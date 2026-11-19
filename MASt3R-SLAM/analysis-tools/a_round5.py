#!/usr/bin/env python3
"""Round-five local exploratory mechanism diagnosis. No pose optimization/inference."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from a_round4 import sha,verify,read
from a_workflow import RAW,GUIDE,PROJECT,relative_errors,quaternion_rotation,label_map,write_json,STATUS
from analyze_round4 import association,category
from audit_round4_results_v1 import audit

VERSION='round5-local-mechanism-v1'
AXES=['tx','ty','tz','rx','ry','rz','log_scale']
FOCUS=['walking_000090_m1','walking_000060_s2','walking_000120_s2','walking_000210_s3','desk_000060_s1','desk_000060_s2']
REQUIRED=['Xf','Xk','Ck','Qk','valid','valid_meas','meas','K','idx_f2k','initial_relative','initial_world','key_world','output_relative','candidate_relative','a_weight','a_status','a_evidence','a_error','a_uv']

def paths(g):
    return dict(g=g,cap=g/'received/a_quality_20261109',res=g/'received/round4_20261110/round4_batch_01',labels=g/'analysis/round4_supplement_v2/labels',scores=g/'analysis/round4_scores_01/scores.json',audit=g/'analysis/round4_supplement_v2/received_audit_v1/audit.json')

def sources(p):
    froot=RAW/'target/MASt3R-SLAM-A/releases/a001/source'
    capture=read(p['cap']/'walking_capture_01/capture.json');out={}
    for name in ['tracker','geometry','reliability','nonlinear_optimizer']:
        rel=f'mast3r_slam/{name}.py';f=froot/rel
        if not f.exists():f=PROJECT/'work/mast3r_a_dev/src'/rel
        if sha(f)!=capture['source_hashes'][rel]:raise ValueError('Frozen solver source hash mismatch: '+str(f))
        out[name]=dict(path=str(f),sha256=sha(f))
    return out

def verify_inputs(g):
    p=paths(g);src=sources(p)
    for d in [p['res'],p['labels']]+[p['cap']/(s+'_capture_01') for s in ['walking','desk']]:verify(d)
    original=read(p['audit'])
    if original['failures'] or original['identity']['results_manifest_sha256']!=sha(p['res']/'files.json') or original['identity']['bundle_manifest_sha256']!=sha(p['labels']/'files.json'):raise ValueError('Round4 independent audit identity mismatch')
    with tempfile.TemporaryDirectory(prefix='a_round5_inputaudit_') as td:
        actual=audit(p['labels'],p['res'],p['cap'],Path(td)/'audit')
        if actual!=original:raise ValueError('Round4 audit recomputation mismatch')
    s=read(p['scores'])
    if s['missing_jobs'] or any(r['failures'] for r in s['rows']) or s['results_sha256']!=sha(p['res']/'files.json') or s['bundle_sha256']!=sha(p['labels']/'files.json'):raise ValueError('Round4 scores identity mismatch')
    watched=[p['scores'],p['audit'],p['labels']/'files.json',p['labels']/'regions.json',p['labels']/'cases.json',p['labels']/'freeze.json',p['res']/'files.json',p['res']/'round.json',p['res']/'identity.json']
    for seq in ['walking','desk']:
        root=p['cap']/(seq+'_capture_01');watched += [root/'files.json',root/'capture.json']
        gt=RAW/'target/datasets/tum'/('rgbd_dataset_freiburg3_walking_xyz' if seq=='walking' else 'rgbd_dataset_freiburg1_desk')/'groundtruth.txt';watched.append(gt)
        if sha(gt)!=s['GT_hashes'][seq]:raise ValueError('GT differs from round4 evaluation source')
        archive=p['res'].parent/('round4_'+('preflight' if seq=='walking' else 'batch')+'_01.tar.gz')
        if sha(archive)!=archive.with_suffix('.gz.sha256').read_text().split()[0]:raise ValueError('Archive checksum mismatch')
        watched.append(archive)
    return dict(version=VERSION,sources=src,input_sha256={str(f.resolve()):sha(f) for f in watched},tools_sha256={str(f.resolve()):sha(f) for f in [Path(__file__),GUIDE/'tools/a_workflow.py',GUIDE/'tools/analyze_round4.py',GUIDE/'tools/a_round4.py',GUIDE/'tools/audit_round4_results_v1.py']},audit=original['status'],exploratory=True,GT_already_public=True,independent_generalization_data=False)

def native_matrix(data):
    x=np.asarray(data,dtype=np.float64).reshape(8)
    if not np.isfinite(x).all() or x[7]<=0:raise ValueError('Invalid native Sim3 data')
    T=np.eye(4);T[:3,:3]=x[7]*quaternion_rotation(x[3:7]);T[:3,3]=x[:3];return T

def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def sim3_log(T):
    """Coordinate conversion only: no normal-equation or pose optimization."""
    T=np.asarray(T,dtype=np.float64).reshape(4,4);s=np.cbrt(np.linalg.det(T[:3,:3]));R=T[:3,:3]/s
    u,_,vt=np.linalg.svd(R);R=u@vt;v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2
    n=np.linalg.norm(v);theta=np.arctan2(n,np.clip((np.trace(R)-1)/2,-1,1));omega=v*(theta/n) if n>1e-12 else v
    sigma=np.log(s);A=sigma*np.eye(3)+skew(omega);V=np.eye(3);term=np.eye(3)
    for k in range(1,45):term=term@A/(k+1);V+=term
    t=np.linalg.solve(V,T[:3,3])
    return np.r_[t,omega,sigma]

def pose_change(before,after):
    delta=sim3_log(after@np.linalg.inv(before))
    return dict(left_sim3_log=delta.tolist(),translation_canonical_norm=float(np.linalg.norm(delta[:3])),rotation_deg=float(np.degrees(np.linalg.norm(delta[3:6]))),log_scale=float(delta[6]),meaning='change magnitude, not GT error; translation is canonical scale, not metres')

def geometry(z,m,T,dtype=np.float64):
    X=np.asarray(z['Xf'],dtype=dtype);K=np.asarray(z['K'],dtype=dtype);T=np.asarray(T,dtype=dtype)
    P=X@T[:3,:3].T+T[:3,3];uvz=P@K.T
    with np.errstate(divide='ignore',invalid='ignore'):uv=uvz[:,:2]/uvz[:,2:3]
    h,w=m['size'];cfg=m['tracking_config'];b=cfg['pixel_border'];depth=P[:,2];goodz=depth>cfg['depth_eps']
    validproj=(uv[:,0]>b)&(uv[:,0]<w-1-b)&(uv[:,1]>b)&(uv[:,1]<h-1-b)&goodz
    logz=np.zeros_like(depth);logz[goodz]=np.log(depth[goodz]);r=np.asarray(z['meas'],dtype=dtype)-np.c_[uv,logz]
    active=z['valid'].ravel()&z['valid_meas'].ravel()&validproj
    # Frozen act_Sim3 left perturbation [translation,rotation,log-scale].
    J=np.zeros((len(P),3,7),dtype=dtype);sel=np.flatnonzero(active);ps=P[sel];xx,yy,zz=ps.T
    D=np.zeros((len(sel),3,3),dtype=dtype);D[:,0,0]=K[0,0]/zz;D[:,0,2]=-K[0,0]*xx/zz**2;D[:,1,1]=K[1,1]/zz;D[:,1,2]=-K[1,1]*yy/zz**2;D[:,2,2]=1/zz
    W=np.zeros((len(sel),3,3),dtype=dtype);W[:,0,1]=-zz;W[:,0,2]=yy;W[:,1,0]=zz;W[:,1,2]=-xx;W[:,2,0]=-yy;W[:,2,1]=xx
    act=np.concatenate([np.broadcast_to(np.eye(3,dtype=dtype),(len(sel),3,3)),-W,ps[:,:,None]],axis=2);J[sel]=-np.einsum('nij,njk->nik',D,act)
    return dict(P=P,uv=uv,r=r,projected=validproj,active=active,J=J)

def huber_weights(e,k):
    e=np.asarray(e);out=np.ones_like(e);mask=np.abs(e)>=k;out[mask]=k/np.abs(e[mask]);return out

def coefficients(z,m,geo):
    sigma=np.array([m['tracking_config']['sigma_pixel']]*2+[m['tracking_config']['sigma_depth']]);active=geo['active']
    raw=np.zeros_like(geo['r']);raw[active]=geo['r'][active];q=z['Qk'].ravel().astype(float);base=np.zeros_like(raw);base[active]=q[active,None]/sigma**2
    e=np.sqrt(base)*raw;h=huber_weights(e,m['tracking_config']['huber']);qh=base*h;w=z['a_weight'].ravel().astype(float)
    rho=np.where(abs(e)<=m['tracking_config']['huber'],.5*e**2,m['tracking_config']['huber']*(abs(e)-.5*m['tracking_config']['huber']))
    return dict(raw=raw,e=e,h=h,Q=base,QH=qh,A=qh*w[:,None],rho=rho)

def normal(J,r,M,selected):
    jj=J[selected];mm=M[selected];rr=r[selected];H=np.einsum('nci,nc,ncj->ij',jj,mm,jj,optimize=True);g=-np.einsum('nci,nc,nc->i',jj,mm,rr,optimize=True);return H,g

def quantile(x):
    x=np.asarray(x).reshape(-1);finite=x[np.isfinite(x)]
    return dict(count=int(len(x)),finite=int(len(finite)),nonfinite=int(len(x)-len(finite)),min_p10_p50_p90_max=np.quantile(finite,[0,.1,.5,.9,1]).tolist() if len(finite) else None)

def sum_components(x,s):return np.sum(x[s],axis=0,dtype=np.float64).tolist()

def read_frame(p,seq,frame):
    f=p['cap']/(seq+'_capture_01')/'frames'/f'{frame:06d}';m=read(f/'metadata.json')
    if sha(f/'inputs.npz')!=m['inputs_sha256']:raise ValueError('Input NPZ hash changed')
    with np.load(f/'inputs.npz',allow_pickle=False) as zp:z={k:np.array(zp[k]) for k in zp.files}
    missing=[k for k in REQUIRED if k not in z]
    if missing:raise ValueError('Required saved fields missing: '+','.join(missing))
    return f,m,z

def selected_replays(p):
    spec=read(p['labels']/'cases.json');mapping={};case_lookup={c['id']:c for c in spec['cases']}
    for job in read(p['res']/'round.json')['results']:
        c=case_lookup[job['case']];r=read(p['res']/job['path']/'replay.json')['results'][0]
        key=(c['sequence'],c['frame']);mapping.setdefault(key,[]).append(dict(case=c,repeat=job['repeat'],path=job['path'],row=r))
    return mapping

def scores_for_poses(matrices,gt,m):
    out={}
    for mode in ['nearest','interpolated']:
        current,ci=association(gt,float(m['current_timestamp']),mode=='interpolated');reference,ri=association(gt,float(m['reference_timestamp']),mode=='interpolated');truth=np.linalg.inv(reference)@current
        values={name:relative_errors(T,truth) for name,T in matrices.items()}
        differences={}
        for name,first,last in [('initial_to_uniform','initial','uniform'),('uniform_to_A_candidate','uniform','A_candidate'),('initial_to_final','initial','final'),('uniform_to_replay_A','uniform','replay_A')]:
            delta={k:None if values[first][k] is None or values[last][k] is None else values[last][k]-values[first][k] for k in ['rotation_deg','translation_direction_deg']};differences[name]=dict(before=first,after=last,delta=delta,category=category(list(delta.values())))
        out[mode]=dict(current=ci,reference=ri,truth_relative=truth.tolist(),errors=values,comparisons=differences)
    return out

def run_poses(g,output):
    if output.exists():raise ValueError('Output exists; inspect check/report, use another output for rerun')
    provenance=verify_inputs(g);p=paths(g);replays=selected_replays(p);rows=[]
    output.mkdir(parents=True)
    try:
        for seq,frame in sorted(replays):
            folder,m,z=read_frame(p,seq,frame);jobs=replays[(seq,frame)];chosen=next(j for j in jobs if j['repeat']==1);branch=chosen['row']['branches']
            for name in ['uniform','A']:
                if any(j['row']['branches'][name]!=branch[name] for j in jobs):raise ValueError('Deduplicated branch mismatch')
            native={name:z[field] for name,field in [('initial','initial_relative'),('A_candidate','candidate_relative'),('final','output_relative')]};native.update(uniform=branch['uniform']['relative_data'],replay_A=branch['A']['relative_data']);matrices={k:native_matrix(v) for k,v in native.items()}
            encoded={k:np.asarray(z[field],dtype=float).reshape(4,4) for k,field in [('initial','initial_relative_matrix'),('final','output_relative_matrix')]};encoded.update(uniform=np.asarray(branch['uniform']['relative_matrix']).reshape(4,4),replay_A=np.asarray(branch['A']['relative_matrix']).reshape(4,4))
            gt=np.loadtxt(RAW/'target/datasets/tum'/('rgbd_dataset_freiburg3_walking_xyz' if seq=='walking' else 'rgbd_dataset_freiburg1_desk')/'groundtruth.txt')
            evaluation=scores_for_poses(matrices,gt,m);represent=scores_for_poses({**matrices,**encoded},gt,m)
            changes={name:pose_change(matrices[a],matrices[b]) for name,a,b in [('initial_to_uniform','initial','uniform'),('uniform_to_A_candidate','uniform','A_candidate'),('initial_to_final','initial','final'),('candidate_to_final','A_candidate','final'),('original_to_replay_A','A_candidate','replay_A')]}
            origin_check=float(np.max(abs(np.linalg.inv(native_matrix(z['key_world']))@native_matrix(z['initial_world'])-matrices['initial'])))
            rows.append(dict(sequence=seq,frame=frame,reference_frame=m['reference_frame'],current_timestamp=m['current_timestamp'],reference_timestamp=m['reference_timestamp'],event=m['event'],initial_solver=m['initial_status'],original_candidate_solver=m['refinement'],uniform_solver=branch['uniform']['solver'],replay_A_solver=branch['A']['solver'],pose_source=dict(NPZ=str(folder/'inputs.npz'),NPZ_sha256=sha(folder/'inputs.npz'),replay=str(p['res']/chosen['path']/'replay.json'),all_region_case_repetitions_deduplicated=len(jobs)),poses_native={k:np.asarray(v).reshape(8).tolist() for k,v in native.items()},matrices={k:v.tolist() for k,v in matrices.items()},native_matrix_vs_saved_matrix_max_abs={k:float(np.max(abs(matrices[k]-v))) for k,v in encoded.items()},key_inverse_world_max_abs=origin_check,scores=evaluation,saved_matrix_representation_sensitivity=represent,pose_changes=changes))
        data=dict(schema=1,round=5,status='poses_complete',provenance=provenance,boundaries=dict(no_inference=True,no_optimization=True,algorithm_unchanged=True,exploratory=True,GT_already_seen=True),definitions=dict(initial='post tracking, pre A; not pre-tracking initializer',uniform='same original post-tracking initial pose; extra budget with additional A weight=1; not complete off SLAM',A_candidate='original pre-acceptance raw bounded weighted candidate, from saved native Sim3',replay_A='fourth-round weighted candidate, audit tolerances apply',final='original A accepted or fallback output, not replayed gate logic',coordinate='T_CkCf = inv(T_WCk) @ T_WCf; current camera to reference camera; GT inv(T_WCk_GT) @ T_WCf_GT',precision='all stages reconstructed from native [t, normalized xyzw quaternion, positive scale] to float64; existing relative_errors does SO3 projection; encoded matrix sensitivity separate'),pose_rows=rows,missing_fields=['Cf and separate valid_match/Cf rejection masks','raw Qff and Qkf factors','source/target descriptors and target sampled descriptors','per-iteration poses/residuals/support/step costs; stopping criterion trigger is not recorded'],regions=[])
        write_json(output/'experiment.json',data)
    except Exception as exc:
        write_json(output/'experiment.json',dict(status='poses_failed',failure=str(exc),provenance=provenance,pose_rows=rows));raise
    return data


def log_checks(z,m,geo,co):
    active=geo['active'];valid=z['valid'].ravel();w=z['a_weight'].ravel();evidence=z['a_evidence'];event=m['event']
    median=float(np.median(np.linalg.norm(geo['r'][active,:2],axis=1)));torch_median=float(np.partition(np.linalg.norm(geo['r'][active,:2],axis=1),(int(active.sum())-1)//2)[(int(active.sum())-1)//2]);cost=float(np.sum(w[:,None]*co['rho']))
    expected=np.ones(len(w));expected[evidence]=np.clip(np.exp(-z['a_error'][evidence].astype(float)/m['a_config']['temperature']),m['a_config']['min_weight'],1)
    return dict(support_count=int(active.sum()),event_support=int(event['support']),support_matches=int(active.sum())==event['support'],event_evidence_matches=int((evidence&valid).sum())==event['evidence'],state_counts_match=all(int((z['a_status']==i).sum())==event['states'][name] for i,name in enumerate(STATUS)),evidence_matches_status=bool(np.array_equal(evidence,z['a_status']==0)),fixed_weight_formula_max_abs=float(np.max(abs(w-expected))),unknown_weight_one=bool((w[~evidence]==1).all()),C_Q_threshold_necessary=bool((z['Ck'].ravel()[valid]>m['tracking_config']['C_conf']).all() and (z['Qk'].ravel()[valid]>m['tracking_config']['Q_conf']).all()),pixel_median_numpy=median,pixel_median_lower_order=torch_median,pixel_log=event['initial_pixel_median'],pixel_log_abs_delta=abs(torch_median-event['initial_pixel_median']),robust_cost=cost,cost_log=event['cost_initial'],cost_relative_delta=abs(cost-event['cost_initial'])/max(1,abs(event['cost_initial'])))

def group_diagnostic(z,m,geo,co,mask,fullH,fullg):
    idx=z['idx_f2k'];h,w=m['size'];selection=mask.reshape(-1)[idx].astype(bool);active=geo['active'];scale=np.sqrt(np.diag(fullH));weight=z['a_weight'].ravel().astype(float)
    def population(s):
        ids=idx[s];unique,repeat=np.unique(ids,return_counts=True);uv=np.c_[ids%w,ids//w];n=int(s.sum())
        return dict(observations=n,unique_source_pixels=len(unique),repeat_factor=float(n/len(unique)) if len(unique) else None,max_multiplicity=int(repeat.max()) if len(repeat) else None,fraction_repeated_observations=float((n-len(unique))/n) if n else None,frame_initial_support_fraction=float(n/max(1,active.sum())) if not (s&~active).any() else None,frame_candidate_fraction=float(n/len(idx)),Ck=quantile(z['Ck'][s]),Qk=quantile(z['Qk'][s]),pixel_residual=quantile(np.linalg.norm(geo['r'][s,:2],axis=1)),component_abs_residual=[quantile(abs(geo['r'][s,i])) for i in range(3)],descriptor_error=quantile(z['a_error'][s]),A_weight=quantile(weight[s]),A_states={name:int(((z['a_status']==i)&s).sum()) for i,name in enumerate(STATUS)},A_evidence=int((z['a_evidence']&s).sum()),A_evidence_fraction=float((z['a_evidence']&s).sum()/n) if n else None,restored_weight_mass=float(np.sum(1-weight[s])),bbox=([int(x) for x in [uv[:,0].min(),uv[:,1].min(),uv[:,0].max(),uv[:,1].max()]] if n else None),occupied_8x8_cells=len(np.unique((ids%w)*8//w+8*((ids//w)*8//h))) if n else 0)
    sel=selection&active;contrib={}
    for name,M in [('Q',co['Q']),('QH',co['QH']),('QHA',co['A'])]:
        H,g=normal(geo['J'],co['raw'],M,sel)
        contrib[name]=dict(effective_information_mass=sum_components(M,sel),irls_residual_energy_components=(.5*np.sum(M[sel]*co['raw'][sel]**2,axis=0)).tolist(),H=H.tolist(),g=g.tolist(),H_diag_fraction_of_full_A=(np.diag(H)/np.diag(fullH)).tolist(),g_normalized=(g/scale).tolist())
    # Huber objective is separate from IRLS b.T@b surrogate used by solver stopping.
    rho=co['rho'];contrib['true_Huber_cost_uniform']=float(rho[sel].sum());contrib['true_Huber_cost_A']=float((rho[sel]*weight[sel,None]).sum())
    H0,g0=normal(geo['J'],co['raw'],co['QH']*(1-weight[:,None]),sel)
    info=population(sel);qenergy=sum(contrib['Q']['irls_residual_energy_components']);henergy=sum(contrib['QH']['irls_residual_energy_components']);aenergy=sum(contrib['QHA']['irls_residual_energy_components'])
    Hnormalized=np.asarray(contrib['QHA']['H'])/(scale[:,None]*scale[None,:]);eig=np.linalg.eigvalsh(Hnormalized)
    return dict(candidate=population(selection),valid_opt=population(selection&z['valid'].ravel()),initial_support=info,eligible=info['observations']>=64 and info['unique_source_pixels']>=64,contributions=contrib,initial_scalar_energy_before_Q=float(.5*np.sum(co['raw'][sel]**2)),Huber_energy_retention=henergy/qenergy if qenergy else None,A_energy_retention_after_Huber=aenergy/henergy if henergy else None,group_normalized_H_eigenvalues=eig.tolist(),restoring_at_fixed_initial=dict(deltaH=H0.tolist(),delta_g=g0.tolist(),delta_g_normalized=(g0/scale).tolist()),source_pixel_position='processed current; no raw-sensor pixel identity')

def saved_endpoint(z,m,T,original_active):
    geo=geometry(z,m,T);co=coefficients(z,m,geo);common=original_active&geo['active'];weight=z['a_weight'].ravel().astype(float);sigma=np.array([m['tracking_config']['sigma_pixel']]*2+[m['tracking_config']['sigma_depth']]);e=geo['r'][common]*np.sqrt(z['Qk'][common])/sigma;k=m['tracking_config']['huber'];rho=np.where(abs(e)<=k,.5*e**2,k*(abs(e)-.5*k))
    return dict(endpoint_support=int(geo['active'].sum()),initial_support_lost=int((original_active&~geo['active']).sum()),initial_support_gained=int((~original_active&geo['active']).sum()),true_A_Huber_cost_on_initial_support=float((weight[common,None]*rho).sum()) if common.sum()==original_active.sum() else None,pixel_median_on_initial_common=float(np.median(np.linalg.norm(geo['r'][common,:2],axis=1))) if common.any() else None,mask=geo['active'])

def run_regions(g,output):
    file=output/'experiment.json';data=read(file)
    if data['status']!='poses_complete':raise ValueError('Regions requires poses_complete; finished/failed output is not overwritten')
    if data['provenance']!=verify_inputs(g):raise ValueError('Input/tool provenance changed since poses stage')
    p=paths(g);annotations=read(p['labels']/'regions.json');replays=selected_replays(p);published=read(p['scores']);published_rows={(r['case'],r['repeat']):r for r in published['rows']};poses={(r['sequence'],r['frame']):r for r in data['pose_rows']};regions=[];frames=[]
    try:
        for sample in annotations['samples']:
            seq,frame=sample['sequence'],sample['frame'];print(f'CPU diagnostic {seq}/{frame}',flush=True);folder,m,z=read_frame(p,seq,frame);pr=poses[(seq,frame)];T=np.asarray(pr['matrices']['initial']);geo=geometry(z,m,T);co=coefficients(z,m,geo);H,gg=normal(geo['J'],co['raw'],co['A'],geo['active']);scale=np.sqrt(np.diag(H));check=log_checks(z,m,geo,co)
            matrixgeo=geometry(z,m,z['initial_relative_matrix'].reshape(4,4));floatgeo=geometry(z,m,z['initial_relative_matrix'].reshape(4,4),np.float32)
            check['native64_vs_matrix64_support_difference']=int((geo['active']!=matrixgeo['active']).sum());check['native64_vs_matrix32_support_difference']=int((geo['active']!=floatgeo['active']).sum())
            endpoints={name:saved_endpoint(z,m,np.asarray(pr['matrices'][name]),geo['active']) for name in ['uniform','A_candidate','replay_A','final']}
            check['final_logged_cost_relative_delta']=abs(endpoints['final']['true_A_Huber_cost_on_initial_support']-m['event']['cost_final'])/max(1,abs(m['event']['cost_final'])) if endpoints['final']['true_A_Huber_cost_on_initial_support'] is not None else None
            if not all(check[k] for k in ['support_matches','event_evidence_matches','state_counts_match','evidence_matches_status','unknown_weight_one','C_Q_threshold_necessary']):raise ValueError('Source/log/mask check failed')
            if check['fixed_weight_formula_max_abs']>2e-7:raise ValueError('Saved A-weight law check failed')
            f=dict(sequence=seq,frame=frame,checks=check,initial_support=int(geo['active'].sum()),full_H=H.tolist(),full_g=gg.tolist(),column_scale=scale.tolist(),full_normalized_H_eigenvalues=np.linalg.eigvalsh(H/(scale[:,None]*scale[None,:])).tolist(),endpoints={k:{kk:vv for kk,vv in v.items() if kk!='mask'} for k,v in endpoints.items()})
            frames.append(f)
            for r in sample['regions']:
                mask=label_map(dict(image_size=m['size'],regions=[r]))>0;diag=group_diagnostic(z,m,geo,co,mask,H,gg);key=f"{seq}_{frame:06d}_{r['id']}";diag.update(id=key,sequence=seq,frame=frame,region=r['id'],label=r['label'],label_basis=r['basis'],certainty=r['certainty'],focus=key in FOCUS)
                idx=z['idx_f2k'];selected=mask.reshape(-1)[idx];diag['endpoint_region_support']={k:int((v['mask']&selected).sum()) for k,v in endpoints.items()}
                job=next((j for j in replays[(seq,frame)] if j['repeat']==1 and ((r['label']=='static' and j['case']['static_region']==r['id']) or (r['label']=='moving' and j['case']['moving_region']==r['id']))),None)
                branch='restore_static' if r['label']=='static' else 'restore_moving' if r['label']=='moving' else None
                if job and branch in job['row']['branches']:
                    b=job['row']['branches'][branch];A=native_matrix(job['row']['branches']['A']['relative_data']);rest=native_matrix(b['relative_data']);delta=sim3_log(rest@np.linalg.inv(A));dg=np.array(diag['restoring_at_fixed_initial']['delta_g_normalized']);dn=scale*delta;cos=float(dg@dn/(np.linalg.norm(dg)*np.linalg.norm(dn))) if np.linalg.norm(dg)*np.linalg.norm(dn)>1e-20 else None
                    endpoint=saved_endpoint(z,m,rest,geo['active']);diag['actual_restore']=dict(status='saved_solved',case=job['case']['id'],source=str(p['res']/job['path']/'replay.json'),pose_change_from_replay_A=pose_change(A,rest),initial_delta_g_vs_actual_normalized_motion_cosine=cos,initial_delta_g_dot_actual_motion=float(dg@dn),endpoint_support=endpoint['endpoint_support'],region_endpoint_support=int((endpoint['mask']&selected).sum()),published_round4={mode:published_rows[(job['case']['id'],1)]['scores'][mode]['branch_minus_A'][branch] for mode in ['nearest','interpolated']},published_categories={mode:published_rows[(job['case']['id'],1)]['scores'][mode]['categories'][branch] for mode in ['nearest','interpolated']},solver=b['solver'])
                else:diag['actual_restore']=dict(status='skipped_no_candidate' if job and diag['candidate']['observations']==0 else 'not_planned',reason='No recorded restored pose; no pose or GT result is fabricated')
                regions.append(diag)
        data.update(status='regions_complete',frames=frames,regions=regions,definitions_contribution=dict(residual='r=meas-[u,v,log z] at common post-tracking initial pose',S0='active*sqrt(Qk)/[sigma_pixel,sigma_pixel,sigma_depth]',Huber='componentwise min(1,k/abs(S0*r)); computed before A',information='M_Q=active Q/sigma^2; M_QH=M_Q*h; M_QHA=M_QH*A_weight',IRLS_energy='0.5 sum M*r^2; solver reported step cost, NOT exact Huber rho',true_Huber='sum w*rho(S0*r); original A acceptance objective, separate from IRLS surrogate',J='-Dproject(P)*[I,-skew(P),P]; left Sim3 perturbation [t canon units,rotation radians,logscale]',normal='H=sum J.T*M*J; g=-sum J.T*M*r; no H/g solve is performed',normalization='D_i=sqrt(diag(full-frame initial H_A)_i); Hnorm=D^-1 H D^-1; gnorm=D^-1 g',alignment='cosine between D^-1 delta_g from restoring at fixed initial and D*log(T_restore*T_replayA^-1); descriptive linearization, not predicted influence or harm',support='saved valid_opt & valid_meas & project_calib(current saved pose); endpoints reconstructed, intermediate iteration masks unavailable'))
        write_json(file,data)
    except Exception as exc:data.update(status='regions_failed',failure=str(exc),frames=frames,regions=regions);write_json(file,data);raise
    return data


def check_output(output):
    data=read(output/'experiment.json')
    if data['status'] not in ['regions_complete','complete']:raise ValueError('Diagnosis not complete')
    if len(data['pose_rows'])!=16 or len(data['regions'])!=43 or len(data['frames'])!=16:raise ValueError('Coverage mismatch')
    for f in data['frames']:
        c=f['checks']
        if not c['support_matches'] or c['native64_vs_matrix32_support_difference'] or c['native64_vs_matrix64_support_difference']:raise ValueError('Mask mismatch')
        if c['pixel_log_abs_delta']>2e-4 or c['cost_relative_delta']>2e-5 or c['final_logged_cost_relative_delta'] is None or c['final_logged_cost_relative_delta']>2e-5:raise ValueError('CPU formula/log crosscheck exceeds declared reconstruction tolerance')
    for row in data['pose_rows']:
        if row['key_inverse_world_max_abs']>1e-6 or max(row['native_matrix_vs_saved_matrix_max_abs'].values())>1e-6:raise ValueError('Coordinate/native matrix mismatch')
    for path,digest in data['provenance']['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen input changed during diagnosis')
    for path,digest in data['provenance']['tools_sha256'].items():
        if sha(path)!=digest:raise ValueError('Diagnosis tool changed; keep this output and use a new output for revised code')
    for row in data['pose_rows']:
        accepted=row['event']['reason']=='accepted'
        if accepted and row['poses_native']['A_candidate']!=row['poses_native']['final']:raise ValueError('Accepted candidate differs from saved final')
        if not accepted and row['poses_native']['initial']!=row['poses_native']['final']:raise ValueError('Fallback final differs from initial')
    return dict(status='passed',poses=16,regions=43,CPU_formula_check_only=True,no_CUDA_replay=True,no_target_needed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['verify','poses','regions','report','check']:
        q=sub.add_parser(name);q.add_argument('--guide',type=Path,default=GUIDE)
        if name!='verify':q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='verify':v=verify_inputs(a.guide);print(json.dumps(dict(status='passed',input_files=len(v['input_sha256']),sources=v['sources']),ensure_ascii=False,indent=2))
    elif a.command=='poses':d=run_poses(a.guide,a.output);print(d['status'],len(d['pose_rows']))
    elif a.command=='regions':d=run_regions(a.guide,a.output);print(d['status'],len(d['regions']))
    elif a.command=='report':
        from report_round5 import report
        report(a.output)
    else:print(json.dumps(check_output(a.output),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
