#!/usr/bin/env python3
"""Target-side a001 capture and fixed-input replay. See the target operation guide.

The original source is read-only. Runtime hooks record existing computations;
they do not recompute descriptors or change A weights. Capture timing is not a
performance benchmark. GPU behavior must be validated on the target machine.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tarfile


SOURCE = Path('/home/c105/work/mast3r_repro/MASt3R-SLAM-A/releases/a001/source')
SEQUENCES = {'desk': 'rgbd_dataset_freiburg1_desk', 'walking': 'rgbd_dataset_freiburg3_walking_xyz'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_source(source, manifest):
    source = Path(source).resolve()
    m = load(manifest)
    if m['release'] != 'a001':
        raise ValueError('Expected frozen a001 manifest')
    for row in m['files'] + m['compatibility_files']:
        path = (source / row['path']).resolve()
        if not path.is_relative_to(source) or sha(path) != row.get('after_sha256', row.get('sha256')):
            raise ValueError('Source mismatch: ' + row['path'])
    paths = list(source.glob('mast3r_slam/*.py')) + list(source.glob('config/*.yaml')) + [source/'main.py']
    return {str(p.relative_to(source)): sha(p) for p in sorted(paths)}


def activate_source(source):
    source = Path(source).resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source/'thirdparty/mast3r'))
    sys.path.insert(0, str(source/'thirdparty/mast3r/dust3r'))


def probe(args):
    hashes = verify_source(args.source, args.manifest)
    activate_source(args.source)
    import torch
    import lietorch
    import mast3r_slam.tracker as tracker
    import mast3r_slam.reliability as reliability
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    pose = lietorch.Sim3.Identity(1, device='cuda:0')
    test = pose.act(torch.tensor([[0., 0., 1.]], device='cuda:0'))
    torch.cuda.synchronize()
    for module in (tracker, reliability):
        if not Path(module.__file__).resolve().is_relative_to(args.source.resolve()):
            raise ValueError('Wrong module origin')
    return dict(status='passed', source=str(args.source.resolve()), source_hashes=hashes,
                python=sys.version, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                sim3_identity_result=test.cpu().tolist(), modules=[tracker.__file__, reliability.__file__])


def capture_worker(args):
    import numpy as np
    from PIL import Image
    activate_source(args.source)
    import torch
    from mast3r_slam import dataloader, reliability
    from mast3r_slam.tracker import FrameTracker
    from mast3r_slam.config import config
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = args.output.resolve()
    chosen = set(map(int, args.frames.split(',')))
    ctx = {}
    timestamps = []
    original_subsample = dataloader.MonocularDataset.subsample

    def subsample(dataset, stride):
        original_subsample(dataset, stride)
        dataset.rgb_files = dataset.rgb_files[:args.limit]
        dataset.timestamps = dataset.timestamps[:args.limit]
        timestamps[:] = list(map(str, dataset.timestamps))
        save(output/'timestamps.json', timestamps)
    dataloader.MonocularDataset.subsample = subsample
    original_track = FrameTracker.track

    def track(tracker, frame):
        ctx['frame'] = frame
        ctx['reference'] = tracker.keyframes.last_keyframe()
        try:
            return original_track(tracker, frame)
        finally:
            ctx.clear()
    FrameTracker.track = track
    original_estimate = reliability.estimate

    def estimate(*a, **kw):
        result = original_estimate(*a, **kw)
        if ctx.get('selected'):
            ctx['result'] = result
        return result
    reliability.estimate = estimate
    original_opt = FrameTracker.opt_pose_calib_sim3

    def opt(tracker, *a, **kw):
        try:
            result = original_opt(tracker, *a, **kw)
            if ctx.get('selected') and kw.get('weights') is not None:
                ctx['candidate_world'] = result[0].data.detach().clone()
                ctx['candidate_relative'] = result[1].data.detach().clone()
                ctx['refinement'] = dict(kw.get('refinement', {}))
            return result
        except Exception as exc:
            if ctx.get('selected') and kw.get('weights') is not None:
                ctx['candidate_error'] = str(exc).splitlines()[0][:300]
            raise
    FrameTracker.opt_pose_calib_sim3 = opt
    original_record = reliability.FrontendReliability.record

    def record(self, frame_id, reason, **stats):
        if ctx.get('selected'):
            ctx['event'] = dict(frame=frame_id, reason=reason, **stats)
        return original_record(self, frame_id, reason, **stats)
    reliability.FrontendReliability.record = record
    original_run = reliability.FrontendReliability.run
    signature = inspect.signature(original_run)

    def run(self, *a, **kw):
        bound = signature.bind(self, *a, **kw).arguments
        frame_id = int(bound['frame_id'])
        ctx['selected'] = frame_id in chosen
        if not ctx['selected']:
            return original_run(self, *a, **kw)
        # Keep all candidates, including initially non-projectable ones.
        names = ('idx_f2k', 'Xf', 'Xk', 'Ck', 'Qk', 'valid', 'meas', 'valid_meas', 'K')
        tensors = {name: bound[name].detach().clone() for name in names}
        for name in ('initial_world', 'key_world', 'initial_relative'):
            tensors[name] = bound[name].data.detach().clone()
            tensors[name+'_matrix'] = bound[name].matrix().detach().clone()
        result = original_run(self, *a, **kw)
        tensors['output_world'], tensors['output_relative'] = (x.data.detach().clone() for x in result)
        tensors['output_relative_matrix'] = result[1].matrix().detach().clone()
        for name in ('candidate_world', 'candidate_relative'):
            if name in ctx:
                tensors[name] = ctx[name]
        for name, value in ctx.get('result', {}).items():
            tensors['a_'+name] = value.detach().clone()
        frame_dir = output/'frames'/f'{frame_id:06d}'
        frame_dir.mkdir(parents=True, exist_ok=False)
        with (frame_dir/'inputs.npz').open('xb') as f:
            np.savez_compressed(f, **{k:v.cpu().numpy() for k,v in tensors.items()})
        for name, frame in [('current', ctx['frame']), ('reference', ctx['reference'])]:
            rgb = np.uint8(np.clip(frame.uimg.cpu().numpy()*255, 0, 255))
            Image.fromarray(rgb).save(frame_dir/(name+'.png'))
        ref_id = int(bound.get('target_frame_id', ctx['reference'].frame_id))
        save(frame_dir/'metadata.json', dict(schema=1, frame=frame_id, reference_frame=ref_id,
            current_timestamp=timestamps[frame_id], reference_timestamp=timestamps[ref_id],
            size=list(map(int, bound['size'])), initial_status=bound['initial_status'],
            tracking_config=dict(bound['tracker'].cfg), a_config=dict(self.cfg), a_mode=self.mode,
            global_config=dict(config), event=ctx.get('event'), refinement=ctx.get('refinement'),
            candidate_error=ctx.get('candidate_error'), inputs_sha256=sha(frame_dir/'inputs.npz'),
            coverage='all candidate observations; initial solver succeeded; A.run reached',
            descriptors_saved=False))
        print('A_CAPTURE '+str(frame_dir), flush=True)
        return result
    reliability.FrontendReliability.run = run
    cfg = output/'capture.yaml'
    cfg.write_text('inherit: "config/a_'+args.mode+'.yaml"\na:\n  save_every: 0\n  max_diagnostics: 0\n')
    os.environ['MAST3R_A_DIAGNOSTICS'] = str(output/'unused_diagnostics')
    sys.argv = [str(args.source/'main.py'), '--dataset', str(args.source/'datasets/tum'/SEQUENCES[args.sequence]),
                '--config', str(cfg), '--no-viz', '--save-as', str(output/'slam')]
    runpy.run_path(str(args.source/'main.py'), run_name='__main__')


def capture(args):
    hashes = verify_source(args.source, args.manifest)
    requested = sorted(set(map(int, args.frames.split(','))))
    if not requested or min(requested) < 1 or max(requested) >= args.limit:
        raise ValueError('Capture frames must be >=1 and < limit')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(schema=1, status='running', source=str(args.source.resolve()), source_hashes=hashes,
                 requested_frames=requested, sequence=SEQUENCES[args.sequence], mode=args.mode,
                 limit=args.limit, tool_sha256=sha(__file__), timing_is_benchmark=False)
    save(args.output/'capture.json', state)
    command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--source', str(args.source.resolve()),
               '--manifest', str(args.manifest.resolve()), '--output', str(args.output),
               '--sequence', args.sequence, '--mode', args.mode, '--frames', args.frames, '--limit', str(args.limit)]
    with (args.output/'run.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['pid'] = child.pid; save(args.output/'capture.json', state)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            code = 130
    recorded = sorted(int(p.name) for p in (args.output/'frames').glob('*') if (p/'metadata.json').is_file())
    state.update(exit_code=code, recorded_frames=recorded,
                 missing_frames=sorted(set(requested)-set(recorded)),
                 status='complete' if code == 0 and recorded == requested else 'incomplete')
    # Reverify that runtime instrumentation did not change the source.
    state['source_unchanged'] = hashes == verify_source(args.source, args.manifest)
    save(args.output/'capture.json', state)
    if state['status'] != 'complete':
        raise ValueError('Capture incomplete; inspect capture.json and run.log (do not overwrite)')
    return state


def replay(args):
    import numpy as np
    capture_root = args.capture.resolve()
    args.output = args.output.resolve()
    if args.labels:
        args.labels = args.labels.resolve()
    capture_meta = load(capture_root/'capture.json')
    labels_meta = None
    if args.labels:
        labels_meta = load(args.labels/'labels.json')
        if labels_meta['capture_sha256'] != sha(capture_root/'capture.json'):
            raise ValueError('Labels belong to a different capture')
        for row in labels_meta['files']:
            path = (args.labels/row['path']).resolve()
            if not path.is_relative_to(args.labels) or sha(path) != row['sha256']:
                raise ValueError('Label file hash mismatch')
    hashes = verify_source(args.source, args.manifest)
    if hashes != capture_meta['source_hashes']:
        raise ValueError('Replay source differs from capture')
    activate_source(args.source)
    import torch
    import lietorch
    from mast3r_slam.tracker import FrameTracker
    from mast3r_slam.config import set_global_config
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    selected = set(map(int,args.frames.split(','))) if args.frames else None
    for folder in sorted((capture_root/'frames').glob('*')):
        if selected is not None and int(folder.name) not in selected:
            continue
        meta = load(folder/'metadata.json')
        if sha(folder/'inputs.npz') != meta['inputs_sha256']:
            raise ValueError('Capture input hash mismatch')
        with np.load(folder/'inputs.npz', allow_pickle=False) as z:
            arrays = {k:np.array(z[k]) for k in z.files}
        if 'candidate_relative' not in arrays or 'a_weight' not in arrays:
            rows.append(dict(frame=meta['frame'], status='not_replayable', reason=meta['event']))
            continue
        set_global_config(meta['global_config'])
        tracker = FrameTracker.__new__(FrameTracker)
        tracker.cfg = meta['tracking_config']
        tensors = {k:torch.as_tensor(v, device='cuda:0') for k,v in arrays.items()}
        pose = lambda name: lietorch.Sim3(tensors[name])
        def solve(weight):
            refinement = dict(max_iters=int(meta['a_config']['max_refine_iters']), max_condition=meta['a_config']['max_condition'])
            _, relative = tracker.opt_pose_calib_sim3(tensors['Xf'], tensors['Xk'], pose('initial_world'),
                pose('key_world'), tensors['Qk'], tensors['valid'], tensors['meas'], tensors['valid_meas'],
                tensors['K'], tuple(meta['size']), weights=weight, refinement=refinement)
            return relative, refinement
        relative, refinement = solve(tensors['a_weight'])
        delta = (relative * pose('candidate_relative').inv()).log()
        maximum = float(delta.abs().max())
        passed = maximum <= args.tolerance and refinement == meta['refinement']
        row = dict(frame=meta['frame'], reference_frame=meta['reference_frame'], status='passed' if passed else 'failed',
                   max_sim3_log_delta=maximum, tolerance=args.tolerance, solver=refinement, branches={})
        if args.intervene:
            if not passed:
                raise ValueError('Replay mismatch; interventions are not valid')
            weights = {'uniform': torch.ones_like(tensors['a_weight']), 'A': tensors['a_weight']}
            if args.labels:
                label_file = args.labels/f"{meta['frame']:06d}.npy"
                labels = np.load(label_file, allow_pickle=False)
                if labels.shape != tuple(meta['size']) or not np.isin(labels, [0,1,2,3]).all():
                    raise ValueError('Labels must be processed-current HxW uint codes 0/1/2/3')
                gathered = torch.as_tensor(labels.reshape(-1), device='cuda:0')[tensors['idx_f2k'].long()]
                for label, code in [('restore_moving',1), ('restore_static',2)]:
                    if not (gathered == code).any():
                        raise ValueError('Required labelled group is empty: '+label)
                    weights[label] = torch.where((gathered == code)[:,None], 1., tensors['a_weight'])
            for name, weight in weights.items():
                try:
                    p, info = solve(weight)
                    row['branches'][name] = dict(relative_data=p.data.cpu().tolist(), relative_matrix=p.matrix().cpu().tolist(), solver=info)
                except (ValueError, RuntimeError) as exc:
                    row['branches'][name] = dict(error=str(exc).splitlines()[0][:300])
        rows.append(row)
    comparable = [r for r in rows if r['status'] != 'not_replayable']
    result = dict(schema=1, capture=str(capture_root), capture_sha256=sha(capture_root/'capture.json'), source_hashes=hashes, tool_sha256=sha(__file__), labels=labels_meta,
        status='passed' if comparable and all(r['status']=='passed' for r in comparable) else 'failed',
        replayable=len(comparable), skipped=len(rows)-len(comparable), results=rows,
        scope='raw bounded-solver candidates, before A acceptance/fallback gates',
        interpretation='No GT scoring yet. Branch outputs do not establish harmfulness. Uniform also uses the extra refinement budget.')
    save(args.output/'replay.json', result)
    if result['status'] != 'passed':
        raise ValueError('Replay validation failed; see replay.json')
    return result


def pack(args):
    root = args.capture.resolve()
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Refuse symlinks in result bundle')
    archive = root.parent/(root.name+'.tar.gz')
    if archive.exists():
        raise ValueError('Archive already exists')
    save(root/'files.json', dict(schema=1, files=[dict(path=str(p.relative_to(root)), size=p.stat().st_size, sha256=sha(p)) for p in files if p != root/'files.json']))
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(root, arcname=root.name)
    archive.with_suffix('.gz.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    return dict(archive=str(archive), sha256=sha(archive))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('probe', 'capture', '_worker', 'replay'):
        q = sub.add_parser(name)
        q.add_argument('--source', type=Path, default=SOURCE)
        q.add_argument('--manifest', type=Path, default=Path(__file__).resolve().with_name('manifest.json'))
        if name in ('capture', '_worker'):
            q.add_argument('--sequence', choices=SEQUENCES, required=True)
            q.add_argument('--mode', choices=['weighted','diagnostic'], default='weighted')
            q.add_argument('--frames', default='30,60,90,120,150,180,210,240')
            q.add_argument('--limit', type=int, default=242)
            q.add_argument('--output', type=Path, required=True)
        if name == 'replay':
            q.add_argument('--capture', type=Path, required=True)
            q.add_argument('--output', type=Path, required=True)
            q.add_argument('--tolerance', type=float, default=1e-5)
            q.add_argument('--intervene', action='store_true')
            q.add_argument('--labels', type=Path)
            q.add_argument('--frames', default='')
    q = sub.add_parser('pack'); q.add_argument('--capture', type=Path, required=True)
    args = p.parse_args()
    if hasattr(args, 'source'):
        args.source = args.source.resolve()
        args.manifest = args.manifest.resolve()
    if args.command == 'replay' and (args.tolerance <= 0 or (args.labels and not args.intervene)):
        p.error('Positive tolerance required; labels require --intervene')
    result = {'probe':probe, 'capture':capture, '_worker':capture_worker, 'replay':replay, 'pack':pack}[args.command](args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
