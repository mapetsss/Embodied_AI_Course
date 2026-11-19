"""Evidence report and derived mechanism checks for a_round5 report."""
from collections import Counter
from pathlib import Path
import numpy as np
from a_round4 import read,sha
from a_workflow import write_json
from a_round5 import FOCUS,AXES,check_output

CN={'improvement':'双改善','degradation':'双退化','metric_conflict':'冲突','unchanged':'不变'}
def num(x):return '—' if x is None else f'{x:.9g}'
def med(p):return p['min_p10_p50_p90_max'][2] if p['min_p10_p50_p90_max'] is not None else None

def report(output):
    output=Path(output);check=check_output(output);p=output/'experiment.json';d=read(p)
    if (output/'report.md').exists():raise ValueError('Report exists; do not overwrite completed diagnosis')
    counts={};sensitivity=[]
    for seq in ['walking','desk']:
        for transition in ['initial_to_uniform','uniform_to_A_candidate','initial_to_final']:
            counts[seq+':'+transition]=dict(Counter(r['scores']['nearest']['comparisons'][transition]['category'] for r in d['pose_rows'] if r['sequence']==seq))
    for r in d['pose_rows']:
        for transition in ['initial_to_uniform','uniform_to_A_candidate','initial_to_final']:
            a=r['scores']['nearest']['comparisons'][transition];b=r['scores']['interpolated']['comparisons'][transition]
            if a['category']!=b['category']:sensitivity.append(dict(sequence=r['sequence'],frame=r['frame'],transition=transition,nearest=a,interpolated=b))
    frames={(f['sequence'],f['frame']):f for f in d['frames']}
    for key,f in frames.items():
        H=np.asarray(f['full_H']);scale=np.asarray(f['column_scale']);Hn=H/(scale[:,None]*scale[None,:]);eig,vec=np.linalg.eigh(Hn);f['normalized_H_eigenvectors_columns']=vec.tolist()
        for r in d['regions']:
            if (r['sequence'],r['frame'])!=key:continue
            Hr=np.asarray(r['contributions']['QHA']['H'])/(scale[:,None]*scale[None,:]);r['geometry_weak_modes']=dict(fraction_along_weakest_two_full_normalized_H_modes=[float(vec[:,i]@Hr@vec[:,i]/eig[i]) for i in range(2)],definition='same initial A linearization, normalized by full H diagonal; eigenvector signs arbitrary; not an observed influence or causal test')
    summary=dict(pose_comparison_categories=counts,GT_association_category_changes=sensitivity,all_16_original_A_accepted=all(r['event']['reason']=='accepted' for r in d['pose_rows']),original_candidate_final_native_exact=all(r['poses_native']['A_candidate']==r['poses_native']['final'] for r in d['pose_rows']),uniform_all_steps_one=all(r['uniform_solver']['steps']==1 for r in d['pose_rows']),A_steps_histogram=dict(Counter(r['original_candidate_solver']['steps'] for r in d['pose_rows'])),CPU_check=check,hypotheses=dict(descriptor_motion_harm='no stable one-to-one relation; descriptors absent, saved cosine errors only',Q_Huber_A='stacked attenuation directly verified, but uniformly reducing it contradicted by desk60',geometry='coordinate/mode-specific H shares exist; no unique causal explanation of GT change',extra_solve_stopping='extra solve changes all inputs; actual A steps differ on 3 desk inputs, pure weight vs stopping not separable there'),decision='No algorithm change. Keep diagnostic mode; diagnose correlated geometry and solver trajectory only if future research requires it.',minimum_modification_recommended=False)
    d.update(status='complete',mechanism_summary=summary,report_tools_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('plot_round5.py').resolve()):sha(Path(__file__).with_name('plot_round5.py'))});write_json(p,d)
    lines=['# 第五轮：本机机制诊断（探索性）','','日期：2026-11-11。复用walking/desk各8组输入及第四轮真实结果。无网络推理、无CUDA重放、无新位姿优化、无A算法/温度/下限/预算修改。第四轮GT已公开，本轮不是盲测；这16组不能作为未来修改的独立泛化验证集。','',
      '## 结论与当前决策','','最明确的发现是：额外求解本身、权重和停止条件都参与了变化。不能把“比uniform好”直接说成“比进入A之前好”。冻结代码中的Q→分量Huber→A叠加削弱已由真实输入和日志交叉检查，但相似削弱既出现在恢复有利组，也出现在恢复不利组；当前没有足够证据选出统一的最小算法修改。建议暂停扩展加权规则，保留诊断用途。','',
      '## 来源、位姿与评价协议','','来源哈希和冻结源码精确路径在experiment.json/provenance。相关tracker/reliability来自原a001接收副本；geometry与nonlinear_optimizer均逐字节匹配采集源码指纹，其中后者从当前源码树读取，不能仅凭目录名将当前精简tracker当作冻结算法。原第四轮339分支、9空候选跳过及72未计划分支的独立审计重新执行并完全匹配。','',
      '- initial：原跟踪求解已经收敛之后、进入A之前。它不是跟踪开始时的初值。','- uniform：从上述initial继续相同额外预算，A额外权重全1，仍保留Q、Huber和测量噪声。它不是off模式完整系统轨迹。','- A_candidate：采集时保存的原始加权候选；replay_A：第四轮A候选，另列原候选与重放的数值差异。','- final：原接受/回退流程输出。本批16组全为accepted且原生candidate/final逐项相同；这批不能评价回退机制。','',
      '统一方向为T_CkCf=inv(T_WCk)@T_WCf（当前相机→参考相机），GT同方向；所有阶段从原生Sim3的[t,xyzw四元数,正scale]重建float64，四元数归一化，复用relative_errors的SO3投影。保存float32矩阵的评价另作表示敏感性记录，不改原scores.json。','',
      '主评价最近GT时间差<=0.02秒，插值GT为敏感性；时间、最近点、差值、插值端点/比例均记录。GT位移<1毫米或预测平移范数<1e-8时方向不评价，保留None。本批16组均可评价。旋转/方向分列，不合分、不输出未经标定的米制位移误差。','',
      '## 一、初始位姿对照（16组去重）','','下表Δ=后阶段GT误差−前阶段GT误差，负值较好。次数仅描述这8组输入，不是独立统计显著性。','',
      '|序列|比较|双改善|双退化|冲突|不变|','|---|---|---:|---:|---:|---:|']
    for key,c in counts.items():lines.append('|'+key.replace(':','|')+f"|{c.get('improvement',0)}|{c.get('degradation',0)}|{c.get('metric_conflict',0)}|{c.get('unchanged',0)}|")
    lines+=['','|序列/帧|initial旋转/方向°|uniform旋转/方向°|原A候选旋转/方向°|final旋转/方向°|initial→uniform Δ旋转/方向°|uniform→A Δ旋转/方向°|initial→final Δ旋转/方向°|uniform/A实际步数|','|---|---|---|---|---|---|---|---|---|']
    for r in d['pose_rows']:
        sc=r['scores']['nearest'];s=sc['errors'];pair=lambda name:'/'.join(num(s[name][k]) for k in ['rotation_deg','translation_direction_deg']);delta=lambda name:'/'.join(num(sc['comparisons'][name]['delta'][k]) for k in ['rotation_deg','translation_direction_deg'])
        lines.append(f"|{r['sequence']}/{r['frame']}|{pair('initial')}|{pair('uniform')}|{pair('A_candidate')}|{pair('final')}|{delta('initial_to_uniform')}|{delta('uniform_to_A_candidate')}|{delta('initial_to_final')}|{r['uniform_solver']['steps']}/{r['original_candidate_solver']['steps']}|")
    lines+=['','uniform全部16组执行1步，A候选13组1步、desk30/60为2步、desk120为3步。冻结check_convergence以旧代价inf启动，第一步rel_dec为NaN，若第一步停止则只能满足delta_norm<0.001；更新在判停前已应用。因此跟踪已收敛不等于额外求解不改变位姿。未保存每步delta/代价，无法对多步A做同一步数反事实比较。','',
      'desk180反例：initial→uniform方向误差约+0.8031°；uniform→A约−0.5778°，final对initial仍约+0.2253°。A抵消了部分额外求解退化，未使原输出优于initial。desk60则initial→uniform已有旋转退化，而A相对uniform又使两项退化；不能将所有变化归为单个区域。','',
      f'GT插值导致的阶段比较类别变化有{len(sensitivity)}项，逐项见experiment.json/mechanism_summary/GT_association_category_changes；原主协议不事后替换。浮点表示敏感性和original→replay_A位姿差分亦单独保存；微小符号变化不视作实际收益。','',
      '![阶段分解](pose_decomposition.png)','','## 二、求解贡献定义与限制','','令r=meas−[u,v,log z]。C仅以Cf>C_conf和Ck>C_conf参与valid_opt筛选，并参与A的目标深度有效性检查，不是连续求解权重；Cf/valid_match分项未保存，所以不能重建所有筛选原因。本批Ck最小约1而C_conf=0，Ck阈值未进一步排除这些候选。Qk=sqrt(Qff[idx]*Qkf)已保存，原两因子未保存。','',
      '初始实际支持=valid_opt & valid_meas & project_calib有效（border=-10、depth_eps=1e-6）。候选包括不参与求解的点；valid_opt不等于实际支持。重建各已保存终点的支持，desk60加权终点比初始新增2观测，其余基线终点未变；没有中间迭代掩码，不能声称全过程不变。','',
      'S0=active*sqrt(Q)/[sigma_pixel,sigma_pixel,sigma_depth]，本批sigma=[1,1,10]；e=S0*r。Huber h=min(1,1.345/|e|)逐u/v/logz分量计算，随后A以sqrt(w)同时乘残差和Jacobian。M_Q=active Q/sigma²；M_QH=M_Q*h；M_QHA=M_QH*w。','',
      '分别记录系数和ΣM、IRLS残差能量0.5ΣM r²、真正Huber代价Σwρ(e)。IRLS能量是tracker每步线性化使用的代价，不等于A接受检查的真正Huber目标。它们降低都不能自动解释为GT改善。','',
      'Jacobian来自冻结左Sim3扰动：J=−Dproject(P)[I,−skew(P),P]，参数顺序tx/ty/tz（canonical长度）、rx/ry/rz（弧度）、logscale。H=ΣJᵀMJ，g=−ΣJᵀMr；本轮不解Hτ=g。跨分量不用裸trace排名，按D_i=sqrt(full H_A对角)归一化，报告同坐标对角份额和全帧弱特征方向份额。这些是共同初始位姿的局部线性化，不代表实际位姿影响或观测危害。','',
      '原生Sim3 log转换中仅解3×3坐标变换方程，未优化位姿。Δg与实际恢复位姿差的归一化夹角是描述性方向比较，恢复还会改变H及后续线性化，不能据此作因果归因。','',
      '## 三、重点案例与反例','','下表Huber保留率为本组初始IRLS残差能量(QH/Q)，A保留率为(QHA/QH)，两者不是信息量份额或GT收益比例。','',
      '|区域|有效/唯一源像素|像素残差中位|Q中位|描述子误差中位(有限数)|A权重中位|Huber保留率|A后保留率|恢复权重总量(实际支持)|恢复后GT类别|','|---|---|---:|---:|---|---:|---:|---:|---:|---|']
    lookup={r['id']:r for r in d['regions']}
    for k in FOCUS:
        r=lookup[k];v=r['initial_support'];a=r['actual_restore'];lines.append(f"|{k}|{v['observations']}/{v['unique_source_pixels']}|{num(med(v['pixel_residual']))}|{num(med(v['Qk']))}|{num(med(v['descriptor_error']))} ({v['descriptor_error']['finite']})|{num(med(v['A_weight']))}|{num(r['Huber_energy_retention'])}|{num(r['A_energy_retention_after_Huber'])}|{num(v['restored_weight_mass'])}|{CN[a['published_categories']['nearest']]}|")
    lines+=['','walking90运动组402观测/333唯一像素，76条无可用描述子证据而权重保持1；其余证据中有较大余弦差。Huber先将该组Q加权IRLS能量降至约9.91%，A再保留约72.10%；但恢复仍两项改善。它证明“动/特征不一致→降权一定有利”在这个局部案例不成立，不能反过来证明该运动组所有对应都正确。','',
      'walking60/s2与120/s2像素残差中位约0.47，Q中位约5.91与15.76，A中位约0.940与0.967。降权轻微、恢复收益很小；描述子误差不大仍会被连续指数规则轻度削弱。210/s3残差更大、Huber能量保留仅约12.76%，A再保留约72.09%，存在叠加削弱与恢复有利同现。','',
      'desk60/s1与s2是关键反例：同为静态，Huber也明显削弱，A分别再保留约48.18%与83.43%能量；恢复却两项退化。s2的5041条观测只有3040个唯一源像素，重复对应更明显；其rz同坐标H份额约10.78%，高于约4.22%的观测占比，几何贡献确有方向差异，但这不证明其约束方向更正确或错误。','',
      'walking90本组同坐标H份额约0.05%—0.22%，很小；恢复影响也很小，不能仅凭点数/权重或初始H就将其称为关键约束。其夹角与其他组不同，也不足以排除其他区域的抵消、残差偏差和停止条件。','',
      '![重点贡献](focus_contributions.png)','','## 四、全部冻结区域总表（43个，不只展示重点）','','同帧运动组去重；静态、运动、不确定和无候选区域全部保留。描述子误差只在可见证据中有定义，nonfinite计数不被改成0。C/Q、逐分量残差、状态/证据、各阶段系数/能量、H/g、候选/valid_opt/实际支持及源像素重复分布的完整值在experiment.json/regions。','',
      '|区域|标签|候选/唯一|valid_opt|实际支持/唯一|支持占比|重复倍数|C/Q中位|像素残差中位|A中位|证据比例|恢复量候选/支持|Huber/A保留率|覆盖8×8格|实际分支/GT类别|','|---|---|---|---:|---|---:|---:|---|---:|---:|---:|---|---|---:|---|']
    for r in d['regions']:
        c=r['candidate'];v=r['valid_opt'];s=r['initial_support'];a=r['actual_restore'];effect=CN[a['published_categories']['nearest']] if a['status']=='saved_solved' else a['status'];lines.append(f"|{r['id']}|{r['label']}|{c['observations']}/{c['unique_source_pixels']}|{v['observations']}|{s['observations']}/{s['unique_source_pixels']}|{num(s['frame_initial_support_fraction'])}|{num(s['repeat_factor'])}|{num(med(s['Ck']))}/{num(med(s['Qk']))}|{num(med(s['pixel_residual']))}|{num(med(s['A_weight']))}|{num(s['A_evidence_fraction'])}|{num(c['restored_weight_mass'])}/{num(s['restored_weight_mass'])}|{num(r['Huber_energy_retention'])}/{num(r['A_energy_retention_after_Huber'])}|{s['occupied_8x8_cells']}|{effect}|")
    lines+=['','## 五、四项假设的判断','','|假设|现有支持|反例/不足|判断|','|---|---|---|---|',
      '|描述子一致性与运动/危害缺乏稳定对应|walking90运动恢复双改善；静态也出现高描述子误差|只有少量稀疏人工区域；没有完整描述子/标签真值|局部反例明确，不能建立稳定分类关系|',
      '|A与Q/Huber共同削弱有用观测|公式和真实系数证实叠加削弱，walking90及210/s3恢复有利|desk60两组同样叠加削弱，恢复却退化|叠加存在，但有用性/危害不能单靠它判断|',
      '|被降权区域提供方向性约束|同坐标H/弱模份额、重复对应和覆盖存在差异|约束含偏差也可能强；局部H不代表GT方向；无中间轨迹|可描述几何差异，不能确认哪条约束造成GT变化|',
      '|变化主要来自额外求解/停止或初值|uniform本身改变16输入；3个desk输入A/等权步数不同；desk180抵消现象|walking60 A相对uniform仍明显改变方向，不能全归于额外步|求解过程是确定的混杂因素，权重与停止纯效应仍未完全分离|','',
      '## 六、交叉检查、缺口与后续停止条件','','10项新增CPU测试通过：原生Sim3/方向、3×3对数坐标、左扰动Jacobian有限差分、Q与sigma、Huber顺序、A同时缩放残差/J、掩码区别、短GT方向、来源/16帧去重、缺字段和输出冲突。该检查只验证CPU公式和数据对应，不是CUDA求解复现。','',
      '16帧初始支持与原事件数量一致；原生float64与保存矩阵float64/float32重建掩码均一致；状态、证据、未知权重1与保存指数权重关系核验。像素中位及A真正Huber初/终代价与原日志存在有限浮点重建误差，逐帧绝对/相对差记录在frames/checks。工程检查界限为像素2e-4、代价相对2e-5，不是GT收益有效性阈值。','',
      '未保存Cf、valid_match分项、Q两因子、源/目标描述子、中间迭代位姿/残差/支持和停止触发原因。只能检查保存的余弦误差→权重映射，不能重算完整评分器或任意新位姿权重；不能给多步A建立同一步数反事实。后续支持目前仅是已存终点的CPU重建。','',
      '标签待人工确认集中于walking90/m1、60/s2、120/s2、210/s3与desk60/s1/s2等重点区域；全部标签仍是助手稀疏画面复核，没有人工签字。本轮未因GT结果改多边形；边界和对应不确定性无法用三次完全相同的重复消除。','',
      '现有资料足以完成本轮本机机制诊断，目标机需求为无。当前不建议实施某项最小算法修改：全局减弱A或对静态一律恢复都被反例限制。若继续研究，先人工复核重点标签、用剩余独立序列预先冻结评价；纯权重/停止效应还需未来另轮保存每步记录或匹配步数对照，届时另写目标机指南。本轮不提供重跑SLAM命令。','',
      'round5_mechanism_01为首次实现输出，保留不覆盖；随后修正了候选集合“支持占比”字段适用范围并加入工具/接受状态验收，以独立02目录生成本报告，未修改第四轮任何结果。','',
      '## 输入、工具与产物指纹','',f"机器结果：experiment.json；入口a_round5.py SHA256 `{d['provenance']['tools_sha256'][str(Path(__file__).with_name('a_round5.py').resolve())]}`。源码/采集/第四轮/GT逐文件指纹见provenance。报告和图表不是新的求解结果。"]
    (output/'report.md').write_text('\n'.join(lines)+'\n');return d
