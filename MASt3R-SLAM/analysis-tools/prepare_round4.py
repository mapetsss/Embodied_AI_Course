import sys,json,shutil,tarfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
G=Path('/home/banana/jushenzhinenggailun/work/MASt3R-SLAM-A_部署与验证指南');sys.path.insert(0,str(G/'tools'));import a_workflow as w;from a_round4 import audit_arrays,sha,write,verify
O=G/'analysis/round4_supplement_v2';O.mkdir();B=O/'labels';B.mkdir();T=G/'analysis/a_quality_tools_v4';T.mkdir()
for n in ['a_target.py','a_target.sh','a_round4.py']:shutil.copy2(G/'tools'/n,T/n)
shutil.copy2(G.parent/'mast3r_a_dev/releases/a001/manifest.json',T/'manifest.json')
p=T/'a_target.py';s=p.read_text();s=s.replace("raise ValueError('Required labelled group is empty: '+label)","continue  # Static-only cases explicitly omit moving branch")
s=s.replace("raise ValueError('Replay mismatch; interventions are not valid')","save(args.output/'replay.json', dict(status='failed', replayable=1, skipped=0, results=[row], reason='A candidate mismatch; no interventions executed'))\n                raise ValueError('Replay mismatch; interventions are not valid')")
p.write_text(s)
# Separate wrapper uses the same audited activation block.
s=(T/'a_target.sh').read_text().replace('a_target.py','a_round4.py');(T/'a_round4.sh').write_text(s)
old=w.read_json(w.DEFAULT_OUTPUT/'regions.json');ns=w.preprocessor(w.RAW)
regions=[];cases=[];captures={};audit=[]
mon={30:[170,140,235,190],60:[250,185,320,235],90:[300,220,350,270],120:[320,245,410,295],150:[290,300,385,340],180:[290,270,385,320],210:[250,165,335,220],240:[300,205,390,250]}
mov={60:[65,115,130,230],90:[470,120,505,265]}
desk={30:[[310,75,420,155],[45,145,100,190]],60:[[130,240,205,300],[420,50,475,120]],90:[[150,140,200,220],[330,270,375,320]],120:[[65,265,160,315],[350,140,420,190]],150:[[100,150,190,240],[365,235,420,280]],180:[[140,155,185,235],[320,300,370,340]],210:[[285,40,360,100],[330,235,390,280]],240:[[310,100,430,175],[50,215,100,265]]}
def rect(x):
 a,b,c,d=x;return [[a,b],[c,b],[c,d],[a,d]]
for seq in ['walking','desk']:
 C=G/f'received/a_quality_20261109/{seq}_capture_01';verify(C);captures[seq]=sha(C/'capture.json')
 for f in sorted((C/'frames').iterdir()):
  m=w.read_json(f/'metadata.json');frame=m['frame'];z=dict(np.load(f/'inputs.npz',allow_pickle=False));sample=next(x for x in old['samples'] if x['frame']==frame and seq in x['sequence'])
  rgb,K=w.process_rgb(sample['current_rgb'],sample['sequence'],ns)
  assert np.array_equal(rgb,np.asarray(Image.open(f/'current.png')));assert abs(K-z['K']).max()<1e-4
  rs=[]
  def add(id,label,poly,basis,certainty='high'):
   r=dict(id=id,label=label,polygon=poly,basis=basis,certainty=certainty,review_status='assistant_visual_review',limitations='Sparse temporal views; no dense motion ground truth');rs.append(r);return r
  if seq=='walking':
   add('s1','static',rect([150,10,250,38]),'Ceiling beam/roof structure fixed across reference/current and adjacent available views; excludes people')
   add('s2','static',rect(mon[frame]),'Fixed monitor/desk/chair structure, checked in reference/current; frame210 includes chair back and monitor edge, not monitor interior alone')
   for prior in sample['regions']:
    if prior['label']=='static':add('s3','static',prior['polygon'],'Retained original static polygon; rechecked fixed scene structure in paired images')
   if frame in mov:add('m1','moving',rect(mov[frame]),'Person internal torso: changes position/pose relative to monitor and partition between reference/current and adjacent captured views; not inferred solely from category')
   elif frame in [120,210]:
    for r in sample['regions']:
     if r['label']=='moving':add('m1','moving',r['polygon'],'Retained original region; checked against reference/current and adjacent views showing walking displacement/limb pose', 'moderate')
   if frame in [30,150,180,240]:
    add('u1','uncertain',rect({30:[355,85,405,160],150:[0,180,12,230],180:[400,145,420,165],240:[12,140,28,195]}[frame]),'Insufficient unambiguous internal moving support: stationary pose, image boundary, small distant person, or unclear displacement; excluded from moving interventions','uncertain')
  else:
   for j,b in enumerate(desk[frame]):add('s'+str(j+1),'static',rect(b),'Desk rigid equipment/table surface; same objects persist in reference/current; static control sequence')
  masks={}
  for r in rs:
   mask=w.label_map(dict(image_size=m['size'],regions=[r]))>0;masks[r['id']]=mask
   st=audit_arrays(z,m,mask);assert st['frame_support']==m['event']['support'];r['audit']=st
   audit.append(dict(sequence=seq,frame=frame,region=r['id'],label=r['label'],**st))
  im=Image.open(f/'current.png').copy();dr=ImageDraw.Draw(im)
  for r in rs:dr.polygon([tuple(x) for x in r['polygon']],outline={'static':'lime','moving':'red','uncertain':'yellow'}[r['label']],width=2);dr.text(tuple(r['polygon'][0]),r['id'],fill='white')
  im.save(O/f'{seq}_{frame:06d}_regions.png')
  regions.append(dict(sequence=seq,frame=frame,size=m['size'],current_sha256=sha(f/'current.png'),reference_sha256=sha(f/'reference.png'),inputs_sha256=sha(f/'inputs.npz'),preprocessing_exact=True,K_max_delta=float(abs(K-z['K']).max()),regions=rs))
  for sr in [r for r in rs if r['label']=='static']:
   cid=f'{seq}_{frame:06d}_{sr["id"]}';L=B/cid;L.mkdir();labels=np.zeros(tuple(m['size']),dtype=np.uint8);labels[masks[sr['id']]]=2
   moving=next((r for r in rs if r['label']=='moving'),None)
   if moving:
    assert not (masks[moving['id']]&masks[sr['id']]).any();labels[masks[moving['id']]]=1
   np.save(L/f'{frame:06d}.npy',labels,allow_pickle=False)
   write(L/'labels.json',dict(capture_sha256=captures[seq],provisional=False,review='assistant_visual_review; see regions.json',files=[dict(path=f'{frame:06d}.npy',sha256=sha(L/f'{frame:06d}.npy'))]))
   cases.append(dict(id=cid,sequence=seq,frame=frame,static_region=sr['id'],moving_region=moving['id'] if moving else None,has_moving=bool(moving),labels=cid,repeat=3,missing_branch_reason=None if moving else 'No confirmed moving region; restore_moving omitted, never fabricated',static_audit=sr['audit'],moving_audit=moving['audit'] if moving else None))
write(B/'regions.json',dict(coordinate_space='captured processed-current integer pixel centers; polygon inclusive',source_pixel='unique idx_f2k value = row-major processed-current pixel y*512+x; not observation slot or raw sensor pixel',mapping='label_current_flat[idx_f2k[k]]',valid_opt='saved valid: valid_match_k & valid_Cf & valid_Ck & valid_Q',initial_support='valid & valid_meas & project_calib(initial_relative.act(Xf)); projection border=-10 depth_eps=1e-6; support changes with iterations',distribution_quantiles=[0,.1,.5,.9,1],samples=regions))
write(B/'cases.json',dict(schema=1,captures=captures,cases=cases,repeats=3,freeze='Before any new branch GT result; historical 120/210 results already known; no selection by new GT'))
write(B/'freeze.json',dict(status='frozen',date='2026-11-10',regions_sha256=sha(B/'regions.json'),cases_sha256=sha(B/'cases.json'),new_GT_viewed=False,supersedes='round4_supplement_v1',revision_reason='Correct frame210 s2 evidence description; retain original static polygons as separate s3 cases; no GT viewed',revision_policy='Create new version with reason; never overwrite this bundle'))
def manifest(root):write(root/'files.json',dict(files=[dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='files.json']))
def pack(root,name):
 manifest(root);archive=root.parent/name
 with tarfile.open(archive,'w:gz') as t:t.add(root,arcname=root.name)
 archive.with_suffix('.gz.sha256').write_text(sha(archive)+'  '+archive.name+'\n');return sha(archive)
print('labels',pack(B,'round4-labels-v2.tar.gz'));print('tools',pack(T,'a-quality-tools-v4.tar.gz'))
write(O/'experiment.json',dict(status='local_prepared_waiting_target',cases=len(cases),jobs=len(cases)*3,preflight_inputs=16,confirmed_motion_frames=[60,90,120,210],eligible_motion_frames=sorted(set(x['frame'] for x in audit if x['sequence']=='walking' and x['label']=='moving' and x['eligible'])),cuda_validated=False,originals_unchanged=True,audit=audit))
