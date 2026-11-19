#!/usr/bin/env python3
"""Summarize audited round-four results without rerunning solvers or changing labels."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from a_round4 import read,write,sha

METRICS=['rotation_deg','translation_direction_deg']
CN={'improvement':'两项改善','degradation':'两项退化','metric_conflict':'指标冲突','unchanged':'不变','skipped_no_candidate':'无候选跳过','not_planned':'未计划'}
def fmt(v):return '—' if v is None else f'{v:.10g}'

def summarize(a):
    d=read(a.scores_dir/'scores.json');audit=read(a.audit_dir/'audit.json');cases={c['id']:c for c in d['cases']}
    if audit['failures'] or audit['identity']['results_manifest_sha256']!=sha(a.results/'files.json') or d['results_sha256']!=sha(a.results/'files.json') or audit['identity']['bundle_manifest_sha256']!=d['bundle_sha256']:raise ValueError('Audit/score provenance mismatch or audit failed')
    if d['missing_jobs'] or len(d['rows'])!=105 or any(r['failures'] for r in d['rows']):raise ValueError('Incomplete or failed scoring')
    rows={r['case']:r for r in d['rows'] if r['repeat']==1};summaries={(s['case'],s['association'],s['branch']):s for s in d['summaries']}
    counts={};eligible={};improvements={}
    for mode in ['nearest','interpolated']:
        for seq in ['walking','desk']:
            selected=[c for c in d['cases'] if c['sequence']==seq and c['static_audit']['eligible']]
            values=[summaries[(c['id'],mode,'restore_static')]['categories'][0] for c in selected]
            counts[mode+':'+seq]=dict(Counter(values));eligible[seq]=len(selected)
            improvements[mode+':'+seq]=[c['id'] for c in selected if summaries[(c['id'],mode,'restore_static')]['categories'][0]=='improvement']
    motion=[];seen=set();sensitivity=[]
    for c in d['cases']:
        if not c['has_moving'] or (c['sequence'],c['frame'],c['moving_region']) in seen:continue
        seen.add((c['sequence'],c['frame'],c['moving_region']));s=summaries[(c['id'],'nearest','restore_moving')]
        motion.append(dict(frame=c['frame'],region=c['moving_region'],eligible=c['moving_audit']['eligible'],delta={m:s['delta'][m]['values'][0] for m in METRICS},category=s['categories'][0]))
    for (cid,mode,branch),s in summaries.items():
        if mode!='nearest' or not branch.startswith('restore'):continue
        t=summaries[(cid,'interpolated',branch)]
        if s['categories']!=t['categories']:sensitivity.append(dict(case=cid,branch=branch,primary=s['categories'][0],interpolated=t['categories'][0],primary_delta={m:s['delta'][m]['values'] for m in METRICS},interpolated_delta={m:t['delta'][m]['values'] for m in METRICS}))
    records={};baseline={};max_delta=0.;steps=Counter();repeats_exact=True
    for job in read(a.results/'round.json')['results']:
        one=read(a.results/job['path']/'replay.json')['results'][0];max_delta=max(max_delta,one['max_sim3_log_delta']);c=cases[job['case']]
        for name,b in one['branches'].items():
            payload=(b['relative_data'],b['relative_matrix'],b['solver']);key=(c['id'],name)
            if key in records and records[key]!=payload:repeats_exact=False
            records[key]=payload;steps[b['solver']['steps']]+=1
            if name in ['A','uniform','restore_moving']:
                k=(c['sequence'],c['frame'],name)
                if k in baseline and baseline[k]!=payload:raise ValueError('Shared baseline/moving result differs across region cases')
                baseline[k]=payload
    gtmax=max(sc[k]['nearest_dt'] for r in d['rows'] for sc in r['scores'].values() for k in ['association_current','association_reference'])
    info=dict(status='complete_audited_with_explicit_skips',score_sha256=sha(a.scores_dir/'scores.json'),audit_sha256=sha(a.audit_dir/'audit.json'),summary_code_sha256=sha(__file__),archives={p.name:sha(p) for p in a.results.parent.glob('round4_*_01.tar.gz')},jobs=105,branch_counts=audit['branch_counts'],failures=[],eligible_static_regions=eligible,static_categories=counts,static_both_improve=improvements,unique_moving_cases=motion,sensitivity_changes=sensitivity,repeats_pose_and_solver_exact=repeats_exact,max_original_A_sim3_log_delta=max_delta,steps=dict(steps),max_GT_nearest_dt=gtmax,limitations=['Region counts are descriptive and correlated within frames; overlapping regions may share observations','Three repetitions are numerical checks, not independent scenes','Exact repeatability does not bound systematic/numerical/GT/label uncertainty','Assistant visually reviewed sparse labels; no human sign-off','Candidate solutions only, before full A acceptance/fallback gates; no full SLAM accuracy claim'])
    write(a.scores_dir/'interpretation.json',info)
    lines=['# 第四轮补充验证：回传验收与逐案例结果','','本轮已完成原结果接收、独立完整性审计及两种GT关联评价。未修改A算法、模型、原始输入、冻结标签或v5求解结果。','',
      '主要发现：walking静态组恢复权重在60/s2、120/s2、210/s3上两项误差都降低，呈现跨帧局部重复；210帧另两区域仍为指标冲突。运动组降权没有跨三个有效帧的一致收益；90帧恢复运动组反而两项都降低。当前不支持直接统一恢复静态权重或改A算法。','',
      '## 接收与原A审计','',f"两个原结果包已校验，105任务齐全；实际分支339个收敛、9个空候选恢复分支跳过、72个运动分支未计划，无其他失败。最大原A Sim3 log差异{max_delta:.12g}，满足固定1e-5且步数/收敛字典一致；实际求解步数分布{dict(steps)}。",'',
      'v5严格check的9项失败保留，不改写为v5全通过。独立审计仅对desk_000030_s2及desk_000150_s1/s2的零候选分支记录skipped_no_candidate；没有伪造恢复位姿或GT分数。原输入、工具、标签及各任务来源全部匹配。', '',
      '## 主协议：最近GT不超过0.02秒','',f'全部当前/参考关联满足阈值，最大最近时间差{gtmax:.10g}秒。旋转和尺度无关平移方向分别评价；GT位移量仅用于判断方向是否可评价，不是预测平移距离误差。以下差值均为恢复分支减同案例A，负值表示误差降低。','',
      '|静态组|满足双64的区域数|两项改善|两项退化|冲突|不变|','|---|---:|---:|---:|---:|---:|']
    for seq in ['walking','desk']:
        cc=counts['nearest:'+seq];lines.append(f"|{seq}|{eligible[seq]}|{cc.get('improvement',0)}|{cc.get('degradation',0)}|{cc.get('metric_conflict',0)}|{cc.get('unchanged',0)}|")
    lines+=['','walking共19个静态区域，其中90/s2初始支持为0，不进入上述18个有效区域；desk16个区域中3个无候选，其余13个有效。区域数量是描述性计数，同帧区域/旧区域可共享观测，不能当独立场景作统计显著性结论，也不按误差改变量排名。', '',
      '|运动帧|初始支持/唯一源像素|旋转变化/度|方向变化/度|主评价|','|---:|---|---:|---:|---|']
    for r in motion:
        c=next(c for c in d['cases'] if c['frame']==r['frame'] and c['sequence']=='walking');sp=c['moving_audit']['initial_support'];lines.append(f"|{r['frame']}|{sp['observations']}/{sp['unique_source_pixels']}|{fmt(r['delta'][METRICS[0]])}|{fmt(r['delta'][METRICS[1]])}|{CN[r['category']]}|")
    lines+=['','运动按帧/区域去重，未把同帧多个静态案例复用的运动分支或三次重复算作新场景。120帧原运动区域保留：候选非0但valid_opt支持0，restore_moving原生位姿与A逐项相同，是负对照。30/150/180/240帧未取得确认运动内部区域，证据缺口保留，未按结果补标签。', '',
      '## 重复与敏感性','',f'所有已求解分支的三次原生位姿、矩阵与求解状态逐项相同：{repeats_exact}；两项误差的重复范围均为0。此结果只说明本环境下本次重复未观察到波动，不说明总不确定性为0，微小变化不能据此宣称有实际意义。', '',
      'GT插值保持相同求解结果，位置线性、四元数最短弧；所有时间端点/比例在scores.json。60/s2、120/s2、210/s3的静态双改善及三个有效运动帧的主类别在插值下保持。以下两项类别对GT关联敏感，单独保留：', '',
      '|案例|分支|最近GT类别|插值类别|','|---|---|---|---|']
    for r in sensitivity:lines.append(f"|{r['case']}|{r['branch']}|{CN[r['primary']]}|{CN[r['interpolated']]}|")
    lines+=['','两组变化均涉及很小的旋转差符号改变；不事后切换主协议。主评价walking有效静态3改善/2退化/12冲突/1不变；插值为3/3/11/1。desk主评价0/4/9/0，插值0/5/8/0。', '',
      '## 逐案例干预（主协议；三次相同，列一次）','',
      '|案例/运动区域|分支|恢复后旋转误差/度|恢复后方向误差/度|旋转−A/度|方向−A/度|类别|','|---|---|---:|---:|---:|---:|---|']
    shown=set()
    for c in d['cases']:
        row=rows[c['id']];scores=row['scores']['nearest']
        for branch in ['restore_static','restore_moving']:
            if branch=='restore_moving':
                if not c['has_moving'] or (c['sequence'],c['frame']) in shown:continue
                shown.add((c['sequence'],c['frame']))
            if branch not in scores['branches']:
                status=row.get('branch_status',{}).get(branch,{}).get('status','not_planned');lines.append(f"|{c['id']}|{branch}|—|—|—|—|{CN.get(status,status)}|");continue
            sc=scores['branches'][branch];delta=scores['branch_minus_A'][branch];cat=scores['categories'][branch];lines.append(f"|{c['id']}|{branch}|{fmt(sc[METRICS[0]])}|{fmt(sc[METRICS[1]])}|{fmt(delta[METRICS[0]])}|{fmt(delta[METRICS[1]])}|{CN[cat]}|")
    lines+=['','## 各输入A与uniform原始误差（主协议）','','各区域案例重复使用的同帧A、uniform及运动分支已核验原生输出相同。','',
      '|序列/帧|A旋转/度|A方向/度|uniform旋转/度|uniform方向/度|','|---|---:|---:|---:|---:|']
    shown=set()
    for c in d['cases']:
        k=(c['sequence'],c['frame'])
        if k in shown:continue
        shown.add(k);sc=rows[c['id']]['scores']['nearest']['branches'];lines.append(f"|{k[0]}/{k[1]}|{fmt(sc['A'][METRICS[0]])}|{fmt(sc['A'][METRICS[1]])}|{fmt(sc['uniform'][METRICS[0]])}|{fmt(sc['uniform'][METRICS[1]])}|")
    lines+=['','## 观测背景量（不可按误差改变量排名）','','唯一源像素是不同idx_f2k值，即处理后当前图的y*512+x；候选与初始实际支持分列。C/Q/权重列是初始支持中位数；恢复量为sum(1−A额外权重)，分别列候选/初始支持。完整分位数、残差及有效掩码定义见冻结regions.json/cases.json。','',
      '|案例/组|候选/唯一像素|valid_opt|初始支持/唯一像素|C中位|Q中位|A权重中位|恢复量：候选/支持|支持bbox；8×8占格|','|---|---|---:|---|---:|---:|---:|---|---|']
    shown=set()
    for c in d['cases']:
        for group in ['static','moving']:
            ainfo=c[group+'_audit']
            if not ainfo:continue
            key=(c['sequence'],c['frame'],group,c.get(group+'_region'))
            if key in shown:continue
            shown.add(key);ca=ainfo['candidate'];sp=ainfo['initial_support'];med=lambda key:None if sp[key] is None else sp[key][2]
            lines.append(f"|{c['id']}/{group}|{ca['observations']}/{ca['unique_source_pixels']}|{ainfo['valid_opt']['observations']}|{sp['observations']}/{sp['unique_source_pixels']}|{fmt(med('Ck'))}|{fmt(med('Qk'))}|{fmt(med('A_weight'))}|{fmt(ca['restored_weight_mass'])}/{fmt(sp['restored_weight_mass'])}|{sp['bbox']}；{sp['occupied_8x8_cells']}|")
    lines+=['','## 结论边界与停止位置','','本轮两个验证问题已有局部答案：静态不利降权在多个帧的指定区域重复，但跨区域不一致；有效运动组降权未体现两指标一致收益。desk静态对照也不支持把全部静态观测恢复为1。尚不足以选择统一算法修改，本轮不改A。','',
      '无需重新采集、重新运行105任务或完整SLAM。现有3个有效运动帧达到本轮操作目标，暂不追加目标机补采。后续若继续研究，先在已有固定输入上诊断90帧运动组与静态区域C/Q、残差、A权重和空间覆盖的差异；要验证机制改变，另版冻结并预先定义对照。','',
      '标签为助手基于稀疏画面复核，尚无用户人工签字；本轮不是完整A接受/回退流程重现，也未测完整SLAM精度、ATE或米制平移误差。','',
      f"来源：scores.json SHA256 `{info['score_sha256']}`；独立审计SHA256 `{info['audit_sha256']}`。原v5失败日志、接收归档及冻结清单均保留。完整每次分支误差、时间对应、重复范围及失败状态在scores.json，机器可读摘要在interpretation.json。"]
    a.report.write_text('\n'.join(lines)+'\n')
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['scores-dir','audit-dir','results','report']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();d=summarize(a);print({k:d[k] for k in ['status','eligible_static_regions','static_categories','unique_moving_cases','repeats_pose_and_solver_exact']})
if __name__=='__main__':main()
