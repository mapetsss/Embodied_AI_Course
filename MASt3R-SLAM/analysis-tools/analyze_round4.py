#!/usr/bin/env python3
"""Score received round-four candidates, preserving failures and repeat spread."""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from a_round4 import read,write,sha,verify
from a_workflow import relative_errors,quaternion_rotation
from audit_round4_results_v1 import inspect_branches

def association(gt,t,interpolate=False):
    i=int(abs(gt[:,0]-t).argmin()); dt=float(abs(gt[i,0]-t))
    if dt>.02:raise ValueError('GT nearest timestamp exceeds 0.02 s')
    info=dict(requested=t,nearest=float(gt[i,0]),nearest_dt=dt)
    row=gt[i].copy()
    if interpolate:
        j=int(np.searchsorted(gt[:,0],t))
        if j==0 or j==len(gt):raise ValueError('No GT interpolation bracket')
        lo,hi=gt[j-1],gt[j];alpha=float((t-lo[0])/(hi[0]-lo[0]));q0=lo[4:]/np.linalg.norm(lo[4:]);q1=hi[4:]/np.linalg.norm(hi[4:]);dot=float(q0@q1)
        if dot<0:q1=-q1;dot=-dot
        dot=np.clip(dot,-1,1)
        if dot>.9995:q=(1-alpha)*q0+alpha*q1
        else:
            theta=np.arccos(dot);q=(np.sin((1-alpha)*theta)*q0+np.sin(alpha*theta)*q1)/np.sin(theta)
        row[1:4]=(1-alpha)*lo[1:4]+alpha*hi[1:4];row[4:]=q/np.linalg.norm(q)
        info.update(bracket=[float(lo[0]),float(hi[0])],alpha=alpha)
    T=np.eye(4);T[:3,:3]=quaternion_rotation(row[4:]);T[:3,3]=row[1:4];return T,info

def category(d):
    if any(x is None for x in d):return 'unscored'
    if all(x==0 for x in d):return 'unchanged'
    if all(x<=0 for x in d):return 'improvement'
    if all(x>=0 for x in d):return 'degradation'
    return 'metric_conflict'

def analyze(a):
    verify(a.bundle);verify(a.results)
    if a.output.exists():raise ValueError('Analysis output exists')
    cases=read(a.bundle/'cases.json');d=read(a.results/'round.json');lookup={x['id']:x for x in cases['cases']}
    gtpaths={'walking':a.walking_gt,'desk':a.desk_gt};gts={k:np.loadtxt(v) for k,v in gtpaths.items()}
    for gt in gts.values():
        if gt.shape[1]!=8 or not np.isfinite(gt).all() or (np.diff(gt[:,0])<=0).any():raise ValueError('Invalid sorted TUM GT')
    metrics=['rotation_deg','translation_direction_deg'];rows=[];groups=defaultdict(list)
    expected={(c['id'],r) for c in cases['cases'] for r in range(1,4)};actual={(r['case'],r['repeat']) for r in d['results']}
    missing=sorted(expected-actual)
    for job in d['results']:
        c=lookup[job['case']];row=dict(case=c['id'],repeat=job['repeat'],sequence=c['sequence'],frame=c['frame'],status=job['status'],static_region=c['static_region'],moving_region=c['moving_region'],scores={},failures=[])
        p=a.results/job['path']/'replay.json'
        try:
            r=read(p)
            if r.get('capture_sha256')!=cases['captures'][c['sequence']] or r['status']!='passed':raise ValueError('Invalid replay identity/status')
            capture=a.capture_root/(c['sequence']+'_capture_01');verify(capture);m=read(capture/'frames'/f"{c['frame']:06d}"/'metadata.json')
            label=np.load(a.bundle/c['labels']/f"{c['frame']:06d}.npy",allow_pickle=False)
            with np.load(capture/'frames'/f"{c['frame']:06d}"/'inputs.npz',allow_pickle=False) as z:
                gathered=label.reshape(-1)[z['idx_f2k']]
            counts={'restore_static':int((gathered==2).sum())}
            if c['has_moving']:counts['restore_moving']=int((gathered==1).sum())
            for name,count in counts.items():
                field='static_audit' if name=='restore_static' else 'moving_audit'
                if count!=c[field]['candidate']['observations']:raise ValueError('Frozen candidate count mismatch')
            row['branch_status'],branch_failures=inspect_branches(c,r['results'][0]['branches'],counts,int(m['a_config']['max_refine_iters']))
            if branch_failures:row['failures'].extend(branch_failures);row['status']='failed'
            elif any(b['status']=='skipped_no_candidate' for b in row['branch_status'].values()):row['status']='complete_with_explicit_skip'
            for mode in ['nearest','interpolated']:
                cur,ci=association(gts[c['sequence']],float(m['current_timestamp']),mode=='interpolated');ref,ri=association(gts[c['sequence']],float(m['reference_timestamp']),mode=='interpolated');truth=np.linalg.inv(ref)@cur
                scores={};errors={}
                for name,b in r['results'][0]['branches'].items():
                    if 'relative_matrix' not in b or not b['solver'].get('converged'):errors[name]=b;continue
                    if not np.isfinite(b['relative_matrix']).all():errors[name]='nonfinite pose';continue
                    scores[name]=relative_errors(b['relative_matrix'],truth)
                diffs={n:{k:(s[k]-scores['A'][k] if s[k] is not None and scores['A'][k] is not None else None) for k in metrics} for n,s in scores.items() if n!='A' and 'A' in scores}
                if errors: row['status']='failed'; row['failures'].append(dict(association=mode,branches=errors))
                row['scores'][mode]=dict(association_current=ci,association_reference=ri,branches=scores,branch_minus_A=diffs,categories={n:category(list(v.values())) for n,v in diffs.items()},failures=errors)
                for name,v in diffs.items():groups[(c['id'],mode,name)].append(v)
        except (ValueError,KeyError,FileNotFoundError) as exc:row['failures'].append(str(exc));row['status']='failed'
        rows.append(row)
    summaries=[]
    for (cid,mode,branch),vs in groups.items():
        brvals={n:[r['scores'][mode]['branches'][n] for r in rows if r['case']==cid and mode in r['scores'] and n in r['scores'][mode]['branches']] for n in ['A',branch]}
        ranges={n:{k:float(np.ptp([s[k] for s in ss if s[k] is not None])) if any(s[k] is not None for s in ss) else None for k in metrics} for n,ss in brvals.items()}
        summary=dict(case=cid,association=mode,branch=branch,repeats=len(vs),branch_ranges=ranges,delta={},categories=[category(list(v.values())) for v in vs])
        for k in metrics:
            vals=[v[k] for v in vs if v[k] is not None]
            if vals:
                span=(ranges['A'][k] or 0)+(ranges[branch][k] or 0)
                summary['delta'][k]=dict(values=vals,min=min(vals),max=max(vals),range=float(np.ptp(vals)),direction_consistent=all(x<0 for x in vals) or all(x>0 for x in vals) or all(x==0 for x in vals),within_observed_repeat_spread=any(abs(x)<=span for x in vals),observed_spread=span)
        summaries.append(summary)
    cross_frame={}
    for mode in ['nearest','interpolated']:
        for branch in ['restore_moving','restore_static']:
            for metric in metrics:
                signs=defaultdict(set)
                for summary in summaries:
                    if summary['association']!=mode or summary['branch']!=branch:continue
                    c=lookup[summary['case']];values=summary['delta'].get(metric,{}).get('values',[])
                    sign='improvement' if len(values)==3 and all(x<0 for x in values) else 'degradation' if len(values)==3 and all(x>0 for x in values) else 'unchanged' if len(values)==3 and all(x==0 for x in values) else 'mixed_or_incomplete'
                    signs[sign].add((c['sequence'],c['frame']))
                cross_frame[mode+':'+branch+':'+metric]={k:sorted(v) for k,v in signs.items()}
    a.output.mkdir(parents=True)
    write(a.output/'scores.json',dict(status='analysis_complete',results_sha256=sha(a.results/'files.json'),bundle_sha256=sha(a.bundle/'files.json'),analysis_sha256=sha(__file__),GT_hashes={k:sha(v) for k,v in gtpaths.items()},primary='nearest <=0.02 seconds',sensitivity='linear translation + shortest arc quaternion interpolation',missing_jobs=missing,cross_frame_direction=cross_frame,rows=rows,summaries=summaries,cases=cases['cases']))
    lines=['# 第四轮补充验证逐案例分析','','旋转与平移方向分别报告；三次重复不是独立场景。候选重放不代表完整SLAM精度。数值方向分类使用正负号，不作为有效性门槛；接近观测波动时谨慎解释。跨区域不按误差改变量排名。背景量见冻结cases.json及scores.json。',f'缺失任务：{missing}','','|案例|关联|分支|重复|旋转变化范围/度|方向变化范围/度|分类|','|---|---|---|---:|---|---|---|']
    for s in summaries:lines.append(f"|{s['case']}|{s['association']}|{s['branch']}|{s['repeats']}|{s['delta'].get(metrics[0])}|{s['delta'].get(metrics[1])}|{s['categories']}|")
    lines+=['','|案例|重复|关联|分支|旋转误差/度|方向误差/度|','|---|---:|---|---|---:|---:|']
    for r in rows:
        for mode,sc in r['scores'].items():
            for branch,score in sc['branches'].items():lines.append(f"|{r['case']}|{r['repeat']}|{mode}|{branch}|{score['rotation_deg']}|{score['translation_direction_deg']}|")
    lines+=['','|案例|组|初始支持/唯一像素|Qk分布 min/p10/p50/p90/max|A权重分布|恢复权重量|bbox / 8x8格数|','|---|---|---|---|---|---:|---|']
    for c in cases['cases']:
        for name in ['static','moving']:
            audit=c.get(name+'_audit')
            if audit:
                support=audit['initial_support'];lines.append(f"|{c['id']}|{name}|{support['observations']}/{support['unique_source_pixels']}|{support['Qk']}|{support['A_weight']}|{support['restored_weight_mass']}|{support['bbox']} / {support['occupied_8x8_cells']}|")
    lines+=['','|案例|重复|分支|跳过/未计划/失败状态|原因|','|---|---:|---|---|---|']
    for row in rows:
        for branch,info in row.get('branch_status',{}).items():
            if info['status']!='solved':lines.append(f"|{row['case']}|{row['repeat']}|{branch}|{info['status']}|{info.get('reason','')}|")
    lines+=['','跨帧方向索引（同帧多区域可同时出现在改善/退化栏，不是独立场景计数；须结合支持量与重复波动）：',str(cross_frame)]
    for r in rows:
        if r['failures']:lines.append(f"失败 {r['case']} r{r['repeat']}: {r['failures']}")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['bundle','results','capture-root','walking-gt','desk-gt','output']:p.add_argument('--'+n,type=Path,required=True)
    analyze(p.parse_args())
if __name__=='__main__':main()
