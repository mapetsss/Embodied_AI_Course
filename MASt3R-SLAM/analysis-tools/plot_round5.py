#!/usr/bin/env python3
"""Static scientific figures from round-five JSON; no solver calls."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot(output):
    d=json.loads((output/'experiment.json').read_text());files=[output/'pose_decomposition.png',output/'focus_contributions.png']
    if any(p.exists() for p in files):raise ValueError('Figure exists; use a fresh diagnosis directory')
    fig,axs=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    effects=['initial_to_uniform','uniform_to_A_candidate','initial_to_final'];labels=['Extra uniform solve','Weighting incl. stopping','Original final vs initial'];colors=['#5b8ff9','#e6a23c','#5a9d72']
    for col,seq in enumerate(['walking','desk']):
        rows=sorted([r for r in d['pose_rows'] if r['sequence']==seq],key=lambda r:r['frame']);x=np.arange(len(rows))
        for rr,metric in enumerate(['rotation_deg','translation_direction_deg']):
            ax=axs[rr,col]
            for i,(effect,label,color) in enumerate(zip(effects,labels,colors)):
                y=[r['scores']['nearest']['comparisons'][effect]['delta'][metric] for r in rows];ax.bar(x+(i-1)*.25,y,width=.24,label=label,color=color)
            ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,[r['frame'] for r in rows]);ax.set_xlabel('Captured frame');ax.set_ylabel(('Rotation' if rr==0 else 'Translation-direction')+' error change (deg)');ax.set_title(seq+' / negative = lower GT error');ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(fontsize=8);fig.savefig(files[0],dpi=160);plt.close(fig)
    wanted=['walking_000090_m1','walking_000060_s2','walking_000120_s2','walking_000210_s3','desk_000060_s1','desk_000060_s2'];lookup={r['id']:r for r in d['regions']};rows=[lookup[k] for k in wanted];short=['W90 motion','W60 static2','W120 static2','W210 static3','D60 static1','D60 static2']
    fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True);x=np.arange(6)
    axs[0].bar(x-.15,[r['Huber_energy_retention'] for r in rows],.3,label='After Q + Huber / after Q',color='#5b8ff9');axs[0].bar(x+.15,[r['Huber_energy_retention']*r['A_energy_retention_after_Huber'] for r in rows],.3,label='After Q + Huber + A / after Q',color='#e6a23c');axs[0].set_xticks(x,short,rotation=35,ha='right');axs[0].set_ylabel('Initial IRLS residual energy ratio');axs[0].set_ylim(0,1);axs[0].legend(fontsize=8);axs[0].set_title('Energy attenuation is not pose influence or GT quality')
    values=np.array([r['contributions']['QHA']['H_diag_fraction_of_full_A'] for r in rows]);im=axs[1].imshow(values,aspect='auto',vmin=0,vmax=values.max(),cmap='viridis');axs[1].set_yticks(range(6),short);axs[1].set_xticks(range(7),['tx','ty','tz','rx','ry','rz','log s']);axs[1].set_title('Initial group H diagonal / full A H diagonal')
    for i in range(6):
        for j in range(7):axs[1].text(j,i,f'{100*values[i,j]:.1f}%',ha='center',va='center',fontsize=8,color='white' if values[i,j]<.05 else 'black')
    fig.colorbar(im,ax=axs[1],label='Dimensionless same-coordinate share');fig.savefig(files[1],dpi=160);plt.close(fig)
    return [str(p) for p in files]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);print(plot(p.parse_args().output))
