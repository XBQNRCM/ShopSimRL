#!/usr/bin/env python3
"""Plot trainer and systems diagnostics without mixing episode and segment metrics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'artifacts/analysis/wandb/summary.json').read_text())
OUT=ROOT/'artifacts/analysis/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cbd5e1',
 'axes.titleweight':'bold','axes.titlelocation':'left','savefig.facecolor':'white'})

def panel(ax,rows,key,title,color,xkey):
    x=[r[xkey] for r in rows];y=[r.get(key,np.nan) for r in rows]
    ax.plot(x,y,lw=1,color=color,alpha=.35)
    smooth=[np.mean(y[max(0,i-4):i+1]) for i in range(len(y))]
    ax.plot(x,smooth,lw=2,color=color,label='5-point trailing mean')
    for b in ([40.5,80.5,120.5] if xkey=='update' else [20.5,40.5,60.5]):ax.axvline(b,color='#ccd5e1',ls='--',lw=.8)
    ax.set(title=title,xlabel='Optimizer update' if xkey=='update' else 'Rollout step',xlim=(1,x[-1]))
    ax.grid(axis='y',color='#e6eaf1');ax.set_axisbelow(True)

for name,rows,xkey,panels in [
 ('optimization',D['updates'],'update',[
  ('train/entropy_loss','A   Logged token entropy','#245ce5'),('train/ppo_kl','B   PPO sampled log-probability difference','#8246c6'),
  ('train/pg_clipfrac','C   Policy clipping fraction','#007f79'),('train/grad_norm','D   Gradient norm','#bc5b27'),
  ('train/loss','E   Policy training loss','#245ce5'),('train/train_rollout_logprob_abs_diff','F   Train / rollout log-probability gap','#8246c6')]),
 ('systems',D['steps'],'step',[
  ('rollout/response_len/mean','A   Admitted Sample response length (tokens)','#245ce5'),
  ('rollout/truncated_ratio','B   Admitted Sample truncation fraction','#bc5b27'),
  ('perf/rollout_time','C   Rollout wall time (seconds)','#007f79'),('perf/actor_train_tok_per_s','D   Actor training throughput (tokens/s)','#8246c6')])]:
    fig,axes=plt.subplots(len(panels)//2,2,figsize=(12,3.1*(len(panels)//2)),layout='constrained')
    for ax,(key,title,color) in zip(axes.flat,panels):panel(ax,rows,key,title,color,xkey)
    axes.flat[0].legend(frameon=False,fontsize=9)
    fig.supxlabel('Four completed W&B runs; faint = raw values. Sample metrics use token segments, not whole shopping episodes.' if name=='systems' else '160 updates across 80 rollout steps. PPO KL here is not divergence from the initial checkpoint.')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=170)
    plt.close(fig)
print('Rendered optimization and systems figures.')
