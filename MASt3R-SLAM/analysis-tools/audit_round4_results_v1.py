#!/usr/bin/env python3
"""Read-only v5 result audit; record empty-candidate omissions without inventing poses."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from a_round4 import audit_arrays, read, sha, verify, write

VERSION='round4-result-audit-v1'

def inspect_branches(case, branches, counts, max_iters=5):
    planned=['uniform','A','restore_static']+(['restore_moving'] if case['has_moving'] else [])
    statuses={}; failures=[]
    for extra in set(branches)-set(planned):failures.append('Unexpected branch: '+extra)
    for name in planned:
        count=counts.get(name)
        if name not in branches:
            if name.startswith('restore_') and count==0:
                statuses[name]=dict(status='skipped_no_candidate',reason='Frozen target region maps to zero candidate observations; v5 omitted branch',candidate_observations=0,pose_available=False)
            else:
                statuses[name]=dict(status='failed',reason='Required branch missing',candidate_observations=count)
                failures.append('Required nonempty/baseline branch missing: '+name)
            continue
        b=branches[name];solver=b.get('solver',{})
        try:
            matrix=np.asarray(b['relative_matrix']);data=np.asarray(b['relative_data'])
            good=matrix.shape==(1,4,4) and data.shape==(1,8) and np.isfinite(matrix).all() and np.isfinite(data).all() and solver.get('converged') is True and 1<=solver.get('steps',0)<=max_iters and solver.get('max_iters')==max_iters and solver.get('max_condition')==1e7
        except (KeyError,TypeError,ValueError):good=False
        statuses[name]=dict(status='solved' if good else 'failed',candidate_observations=count,solver=solver,pose_available=good)
        if not good:failures.append('Failed/nonconverged/nonfinite/invalid branch: '+name)
    if not case['has_moving']:statuses['restore_moving']=dict(status='not_planned',reason=case['missing_branch_reason'],pose_available=False)
    return statuses,failures

def audit(bundle,results,capture_root,output):
    bundle,results,capture_root,output=map(Path,(bundle,results,capture_root,output))
    verify(bundle);verify(results)
    identity=dict(version=VERSION,tool_sha256=sha(__file__),results_manifest_sha256=sha(results/'files.json'),bundle_manifest_sha256=sha(bundle/'files.json'))
    if output.exists():
        old=read(output/'audit.json')
        if old['identity']!=identity:raise ValueError('Audit output exists with different identity')
        return old
    spec=read(bundle/'cases.json');round_data=read(results/'round.json');run_id=read(results/'identity.json')
    # In this workspace labels resides under round4_supplement_v2; resolve saved v5 separately.
    frozen_tools=Path(__file__).resolve().parents[1]/'analysis/a_quality_tools_v5'
    expected_id=dict(bundle_sha256=sha(bundle/'files.json'),tool_sha256=sha(frozen_tools/'a_round4.py'),target_sha256=sha(frozen_tools/'a_target.py'),phase='batch')
    if run_id!=expected_id or round_data['identity']!=expected_id or round_data['phase']!='batch':raise ValueError('Run identity mismatch')
    cases={c['id']:c for c in spec['cases']};expected={(c['id'],r) for c in spec['cases'] for r in range(1,4)}
    keys=[(j['case'],j['repeat']) for j in round_data['results']]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('Missing/duplicate/unexpected case repetitions')
    capture_meta={};arrays={};metadata={}
    for seq in spec['captures']:
        root=capture_root/(seq+'_capture_01');verify(root)
        if sha(root/'capture.json')!=spec['captures'][seq]:raise ValueError('Capture mismatch')
        capture_meta[seq]=read(root/'capture.json')
    rows=[];failures=[];cache={}
    for job in round_data['results']:
        c=cases[job['case']];row=dict(case=c['id'],repeat=job['repeat'],original_job_status=job['status'],failures=[],branches={})
        try:
            expected_path=f"{c['id']}_r{job['repeat']}"
            if job['path']!=expected_path:raise ValueError('Job path mismatch')
            if job['status']!='passed' or job.get('exit_code')!=0:raise ValueError('Job process did not succeed')
            key=(c['sequence'],c['frame']);root=capture_root/(c['sequence']+'_capture_01');folder=root/'frames'/f"{c['frame']:06d}"
            if key not in arrays:
                metadata[key]=read(folder/'metadata.json')
                if sha(folder/'inputs.npz')!=metadata[key]['inputs_sha256']:raise ValueError('NPZ mismatch')
                with np.load(folder/'inputs.npz',allow_pickle=False) as z:arrays[key]={k:np.array(z[k]) for k in z.files}
            m=metadata[key];z=arrays[key]
            if c['id'] not in cache:
                labelmeta=read(bundle/c['labels']/'labels.json');labels=np.load(bundle/c['labels']/f"{c['frame']:06d}.npy",allow_pickle=False)
                if labelmeta['capture_sha256']!=spec['captures'][c['sequence']] or labels.shape!=tuple(m['size']) or not np.isin(labels,[0,1,2,3]).all():raise ValueError('Invalid label metadata/shape/codes')
                for f in labelmeta['files']:
                    if sha(bundle/c['labels']/f['path'])!=f['sha256']:raise ValueError('Label hash mismatch')
                counts={}
                for branch,code,field in [('restore_static',2,'static_audit'),('restore_moving',1,'moving_audit')]:
                    if branch=='restore_moving' and not c['has_moving']:continue
                    actual=audit_arrays(z,m,labels==code)
                    if actual!=c[field]:raise ValueError('Frozen support audit mismatch')
                    counts[branch]=actual['candidate']['observations']
                cache[c['id']]=(labelmeta,counts)
            labelmeta,counts=cache[c['id']]
            r=read(results/expected_path/'replay.json')
            if r['capture_sha256']!=spec['captures'][c['sequence']] or r['tool_sha256']!=expected_id['target_sha256'] or r['source_hashes']!=capture_meta[c['sequence']]['source_hashes'] or r['labels']!=labelmeta:raise ValueError('Replay provenance mismatch')
            if r['status']!='passed' or r['replayable']!=1 or r['skipped']!=0 or len(r['results'])!=1:raise ValueError('Original A replay failed/skipped')
            one=r['results'][0]
            if one['frame']!=c['frame'] or one['reference_frame']!=m['reference_frame'] or one['status']!='passed' or one['tolerance']!=1e-5 or not np.isfinite(one['max_sim3_log_delta']) or one['max_sim3_log_delta']>1e-5 or one['solver']!=m['refinement']:raise ValueError('A consistency protocol mismatch')
            row['branches'],row['failures']=inspect_branches(c,one['branches'],counts,int(m['a_config']['max_refine_iters']))
            row['static_audit']=c['static_audit'];row['moving_audit']=c['moving_audit']
        except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:row['failures'].append(str(exc))
        if row['failures']:failures.append(dict(case=c['id'],repeat=job['repeat'],reasons=row['failures']))
        row['status']='failed' if row['failures'] else 'complete_with_explicit_skip' if any(b['status']=='skipped_no_candidate' for b in row['branches'].values()) else 'complete'
        rows.append(row)
    counts=Counter(b['status'] for row in rows for b in row['branches'].values())
    result=dict(identity=identity,status='failed' if failures else 'audited_with_explicit_skips' if counts['skipped_no_candidate'] else 'audited_complete',original_v5_check='preserved in original SSH logs; not rewritten by this audit',expected_jobs=len(expected),audited_jobs=len(rows),branch_counts=dict(counts),failures=failures,rows=rows,scope='A candidates and actual branch records only; skipped branches have no pose/GT score; no source result changed')
    output.mkdir(parents=True);write(output/'audit.json',result)
    lines=['# v5原结果的独立完整性审计','','原v5验收失败记录保留。此审计只在冻结区域及原NPZ确认候选数为0时，将缺失恢复分支明确记作skipped_no_candidate；不补位姿、不复制A结果、不修改原数据。',f"状态：{result['status']}；任务{len(rows)}/{len(expected)}；分支计数{dict(counts)}。",'','|案例|重复|分支|状态|候选数|原因|','|---|---:|---|---|---:|---|']
    for row in rows:
        for name,b in row['branches'].items():
            if b['status']!='solved':lines.append(f"|{row['case']}|{row['repeat']}|{name}|{b['status']}|{b.get('candidate_observations')}|{b.get('reason','')}|")
    lines+=['','失败：',json_text(failures)]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result

def json_text(value):
    import json
    return json.dumps(value,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['bundle','results','capture-root','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();d=audit(a.bundle,a.results,a.capture_root,a.output)
    print(json_text({k:d[k] for k in ['status','expected_jobs','audited_jobs','branch_counts','failures']}))
    if d['failures']:raise SystemExit(1)
if __name__=='__main__':main()
