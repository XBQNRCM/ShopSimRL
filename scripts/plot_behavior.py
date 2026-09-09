#!/usr/bin/env python3
"""Standalone scientific figures from frozen numerical behavior summaries."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'artifacts/analysis/behavior/behavior.json').read_text(encoding='utf-8'))
OUT=ROOT/'artifacts/analysis/figures'
BLUE,ORANGE,PURPLE,TEAL='#245ce5','#bc5b27','#8246c6','#007f79'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cbd5e1',
 'axes.titleweight':'bold','axes.titlelocation':'left','savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=170)
    plt.close(fig)

def grid(ax):
    ax.grid(axis='y',color='#e6eaf1');ax.set_axisbelow(True)

fig,axes=plt.subplots(2,2,figsize=(12,7.4),layout='constrained')
for ax,metric,title in zip(axes.flat,['tokens_per_turn','model_steps','searches','unique_products'],
 ['A   Generation per model turn','B   Model turns per episode','C   Executed searches per episode','D   Unique product pages per episode']):
    for outcome,color in [('success',BLUE),('failure',ORANGE)]:
        y=[r[outcome]['metrics'][metric]['mean'] for r in D['rounds']]
        median=[r[outcome]['metrics'][metric]['median'] for r in D['rounds']]
        ax.plot(range(4),y,'o-',color=color,label=outcome.title()+' mean')
        ax.plot(range(4),median,':',color=color,alpha=.7,label=outcome.title()+' median')
    ax.set(title=title,xticks=range(4),xticklabels=['R0','R1','R2','R3'],ylim=(0,None))
    ax.set_ylabel('Re-encoded tokens' if metric=='tokens_per_turn' else 'Count')
    grid(ax)
axes[0,0].legend(frameon=False,ncol=2,fontsize=9)
fig.supxlabel('Generated, scored episodes; changing training tasks and outcome composition. Lines are descriptive.')
save(fig,'behavior_outcomes')

fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for ax,outcome in zip(axes,['success','failure']):
    for i,color in enumerate([BLUE,TEAL,PURPLE,ORANGE]):
        row=next(r for r in D['distributions'] if r['metric']=='model_steps' and r['round']==i and r['outcome']==outcome)
        x=np.array([int(k) for k in row['counts']]); n=np.array(list(row['counts'].values()))
        ax.step(x,np.cumsum(n)/n.sum(),where='post',color=color,label=f'R{i} (n={n.sum():,})')
    ax.set(title=outcome.title()+' episode length distribution',xlabel='Model turns',ylabel='Cumulative fraction',xlim=(0,30),ylim=(0,1.02))
    grid(ax);ax.legend(frameon=False,fontsize=9)
save(fig,'behavior_step_distribution')

fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for ax,metric,title in zip(axes.flat,['tokens_per_turn','generated_tokens','model_steps','searches','invalid_actions','protocol_errors'],
 ['Tokens per turn','Tokens per episode','Model turns per episode','Searches per episode','Invalid environment actions','Protocol parsing errors']):
    for outcome,color in [('all',PURPLE),('success',BLUE),('failure',ORANGE)]:
        vals=[(D['evaluation'][c] if outcome=='all' else D['evaluation_outcomes'][c][outcome])['metrics'][metric]['mean']
              for c in ['val_base','val_20','val_40','val_60','val_80']]
        ax.plot([0,20,40,60,80],vals,'o-',color=color,label=outcome.title())
    ax.set(title=title,xticks=[0,20,40,60,80],ylim=(0,None));grid(ax)
axes[0,0].legend(frameon=False,fontsize=9)
fig.supxlabel('Frozen validation tasks, skill-free evaluation; API completion tokens. Outcome subgroups change with checkpoint.')
save(fig,'behavior_fixed_validation')

fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for ax,keys,title in [(axes[0],['tokens_per_turn','generated_tokens','first_turn_tokens'],'A   Generation cost: Iter80 minus Base'),
                      (axes[1],['model_steps','searches','unique_products','pagination','invalid_actions','protocol_errors'],'B   Interaction: Iter80 minus Base')]:
    names={'tokens_per_turn':'Tokens / turn','generated_tokens':'Tokens / episode','first_turn_tokens':'First-turn tokens',
           'model_steps':'Model turns','searches':'Searches','unique_products':'Unique products','pagination':'Pagination',
           'invalid_actions':'Invalid actions','protocol_errors':'Protocol errors'}
    for j,(pair,color,label) in enumerate([('fixed_val_all',PURPLE,'All 400 tasks'),('fixed_val_both_success',BLUE,'211 tasks successful in both')]):
        for i,k in enumerate(keys):
            r=D['paired'][pair]['metrics'][k];lo,hi=r['ci95'];delta=r['difference']
            ax.errorbar(delta,i+(j-.5)*.2,xerr=[[delta-lo],[hi-delta]],fmt='o',color=color,capsize=3,label=label if i==0 else None)
    ax.axvline(0,color='#94a3b8',ls='--');ax.set(yticks=range(len(keys)),yticklabels=[names[k] for k in keys],title=title)
    ax.invert_yaxis();grid(ax);ax.legend(frameon=False,fontsize=8,loc='lower left')
fig.supxlabel('Task-paired mean differences and 95% percentile intervals; 5,000 resamples. Exploratory, without multiplicity correction.')
save(fig,'behavior_paired_effects')

fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for phase,color,label in [('search_home',BLUE,'Search home'),('search_results',TEAL,'Search results'),('product_detail',PURPLE,'Product detail'),('protocol_repair',ORANGE,'Protocol repair')]:
    rows=[r for r in D['phases'] if r['phase']==phase]
    for ax,key in zip(axes,['mean','p90']):
        ax.plot(range(4),[r[key] for r in rows],'o-',color=color,label=label)
for ax,title in zip(axes,['A   Mean generation length by page phase','B   90th percentile generation length']):
    ax.set(title=title,xticks=range(4),xticklabels=['R0','R1','R2','R3'],ylabel='Re-encoded tokens per model turn',ylim=(0,None));grid(ax)
axes[0].legend(frameon=False,fontsize=9)
fig.supxlabel('Turn-weighted distributions among scored training episodes; phase is the state before the action.')
save(fig,'behavior_phases')
print('Rendered five behavior figures (SVG + PNG).')
