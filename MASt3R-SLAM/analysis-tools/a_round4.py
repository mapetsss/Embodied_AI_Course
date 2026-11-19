#!/usr/bin/env python3
"""Round-four frozen case orchestration; no model inference."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import hashlib
import numpy as np

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n'); t.replace(p)
def verify(root):
    root=Path(root).resolve(); m=read(root/'files.json')
    declared=set()
    for r in m['files']:
        p=(root/r['path']).resolve()
        if not p.is_relative_to(root) or p.is_symlink() or sha(p)!=r['sha256'] or p.stat().st_size!=r['size']: raise ValueError('Payload mismatch: '+str(p))
        if r['path'] in declared: raise ValueError('Duplicate payload')
        declared.add(r['path'])
    if declared!={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='files.json'}: raise ValueError('Missing/undeclared payload')
    return m

def audit_arrays(z,m,mask):
    idx=z['idx_f2k']; h,w=m['size']
    if mask.shape!=(h,w) or idx.shape!=(h*w,) or idx.min()<0 or idx.max()>=h*w: raise ValueError('Index/shape mismatch')
    selected=mask.reshape(-1)[idx].astype(bool)
    T=z['initial_relative_matrix'].reshape(4,4); P=z['Xf']@T[:3,:3].T+T[:3,3]
    p=P@z['K'].T; uv=p[:,:2]/p[:,2:3]; b=m['tracking_config']['pixel_border']
    projected=(uv[:,0]>b)&(uv[:,0]<w-1-b)&(uv[:,1]>b)&(uv[:,1]<h-1-b)&(P[:,2]>m['tracking_config']['depth_eps'])
    valid=z['valid'].reshape(-1); active=valid&z['valid_meas'].reshape(-1)&projected
    pz=np.column_stack([uv,np.log(np.maximum(P[:,2],1e-30))]); residual=z['meas']-pz
    weight=z['a_weight'].reshape(-1)
    def stats(s):
        ids=idx[s]; yy,xx=np.divmod(ids,w)
        def q(a):
            a=np.asarray(a)[s]; a=a[np.isfinite(a)]; return np.quantile(a,[0,.1,.5,.9,1]).tolist() if len(a) else None
        return dict(observations=int(s.sum()),unique_source_pixels=int(len(np.unique(ids))),
          Ck=q(z['Ck'].reshape(-1)),Qk=q(z['Qk'].reshape(-1)),pixel_residual=q(np.linalg.norm(residual[:,:2],axis=1)),log_depth_residual=q(abs(residual[:,2])),A_weight=q(weight),
          restored_weight_mass=float(np.sum(1-weight[s],dtype=np.float64)),bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(ids) else None,
          occupied_8x8_cells=int(len(set(zip(xx*8//w,yy*8//h)))) if len(ids) else 0)
    support=stats(selected&active)
    return dict(candidate=stats(selected),valid_opt=stats(selected&valid),initial_support=support,eligible=support['observations']>=64 and support['unique_source_pixels']>=64,frame_support=int(active.sum()))

def plan(bundle):
    d=read(Path(bundle)/'cases.json')
    return [(c,r) for c in d['cases'] for r in range(1,4)]

def run(a):
    bundle=a.bundle.resolve(); verify(bundle); cases=read(bundle/'cases.json')
    if a.phase=='batch':
        if a.preflight is None: raise ValueError('Batch requires --preflight')
        from types import SimpleNamespace
        check(SimpleNamespace(bundle=bundle,output=a.preflight))
        pre=read(a.preflight/'round.json')
        if pre['phase']!='preflight' or pre['status']!='passed' or pre['identity']['bundle_sha256']!=sha(bundle/'files.json') or pre['identity']['target_sha256']!=sha(Path(__file__).with_name('a_target.py')): raise ValueError('Preflight identity mismatch')
    for seq in ('walking','desk'):
        verify(a.capture_root/(seq+'_capture_01'))
        if sha(a.capture_root/(seq+'_capture_01')/'capture.json')!=cases['captures'][seq]: raise ValueError('Wrong capture')
    identity=dict(bundle_sha256=sha(bundle/'files.json'),tool_sha256=sha(__file__),target_sha256=sha(Path(__file__).with_name('a_target.py')),phase=a.phase)
    if a.output.exists():
        if not a.resume or read(a.output/'identity.json')!=identity: raise ValueError('Output exists or resume identity mismatch')
    else:
        a.output.mkdir(parents=True);write(a.output/'identity.json',identity)
    # One process per case/repetition: each first repeats original A consistency.
    jobs=[(c,r) for c,r in plan(bundle) if a.phase=='batch' or (r==1 and c['id'].endswith('s1'))]
    results=[]
    for c,r in jobs:
        name=f"{c['id']}_r{r}"; dest=a.output/name
        if dest.exists():
            if (dest/'replay.json').exists():
                old=read(dest/'replay.json')
                results.append(dict(case=c['id'],repeat=r,status=old['status'],path=name));continue
            results.append(dict(case=c['id'],repeat=r,status='interrupted',path=name,reason='Incomplete directory retained; use a new output root'));continue
        cmd=[sys.executable,str(Path(__file__).with_name('a_target.py')),'replay','--capture',str(a.capture_root/(c['sequence']+'_capture_01')),'--frames',str(c['frame']),'--output',str(dest)]
        if a.phase=='batch': cmd+=['--intervene','--labels',str(bundle/c['labels'])]
        proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (a.output/(name+'.log')).write_text(proc.stdout)
        results.append(dict(case=c['id'],repeat=r,status='passed' if proc.returncode==0 else 'failed',path=name,exit_code=proc.returncode))
        write(a.output/'round.json',dict(identity=identity,phase=a.phase,results=results,status='running'))
    status='passed' if len(results)==len(jobs) and all(x['status']=='passed' for x in results) else 'failed'
    write(a.output/'round.json',dict(identity=identity,phase=a.phase,results=results,status=status,expected_jobs=len(jobs)))
    if status!='passed': raise ValueError('Some jobs failed/interrupted; inspect round.json and logs')

def check(a):
    verify(a.bundle); cases=read(a.bundle/'cases.json'); d=read(a.output/'round.json')
    expected={(c['id'],r) for c,r in plan(a.bundle) if d['phase']=='batch' or (r==1 and c['id'].endswith('s1'))}
    actual={(x['case'],x['repeat']) for x in d['results']}
    if actual!=expected or len(d['results'])!=len(expected): raise ValueError('Job coverage mismatch')
    failures=[]
    for x in d['results']:
        p=a.output/x['path']/'replay.json'
        if not p.exists(): failures.append(x);continue
        r=read(p)
        c=next(c for c in cases['cases'] if c['id']==x['case'])
        if r.get('capture_sha256')!=cases['captures'][c['sequence']] or r.get('tool_sha256')!=d['identity']['target_sha256']: failures.append(x)
        if r['status']!='passed' or r['replayable']!=1 or r['skipped']!=0: failures.append(x)
        if d['phase']=='batch':
            c=next(c for c in cases['cases'] if c['id']==x['case'])
            branches=r['results'][0]['branches']; required={'uniform','A','restore_static'}|({'restore_moving'} if c['has_moving'] else set())
            if set(branches)!=required or any('relative_matrix' not in b or not b['solver'].get('converged') or not np.isfinite(b['relative_matrix']).all() or not np.isfinite(b['relative_data']).all() for b in branches.values()): failures.append(x)
    print(json.dumps(dict(expected=len(expected),failures=failures,status='passed' if not failures else 'failed'),indent=2))
    if failures: raise ValueError('Incomplete/failed results')

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for name in ['run','check','verify']:
        q=s.add_parser(name);q.add_argument('--bundle',type=Path,required=True)
        if name!='verify':q.add_argument('--output',type=Path,required=True)
        if name=='run':
            q.add_argument('--capture-root',type=Path,required=True);q.add_argument('--phase',choices=['preflight','batch'],required=True);q.add_argument('--resume',action='store_true');q.add_argument('--preflight',type=Path)
    a=p.parse_args()
    if a.command=='run':
        import fcntl
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with (a.output.parent/(a.output.name+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(a)
    elif a.command=='check':check(a)
    else:verify(a.bundle);print('PAYLOAD_OK')
if __name__=='__main__':main()
