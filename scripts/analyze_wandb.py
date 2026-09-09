#!/usr/bin/env python3
"""Join unsampled W&B history on explicit rollout/train steps, then verify local metrics."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/analysis/wandb'

def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csv_write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert len(manifest['runs'])==4
    batches={};updates=[]
    for run in manifest['runs']:
        assert run['primary'] and run['round'] in range(4)
        path=OUT/(run['id']+'.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==run['sha256']
        history=json.loads(path.read_text())['history']
        for row in history:
            if 'train/step' in row:
                train_step=int(row['train/step']);step=train_step//2
                assert (step-1)//20==run['round']
                updates.append({'update':train_step-1,'train_step':train_step,'step':step,'round':run['round'],
                                **{k:v for k,v in row.items() if k.startswith('train/')}})
            elif 'rollout/step' in row:
                step=int(row['rollout/step']);assert (step-1)//20==run['round']
                batch=batches.setdefault(step,{'step':step,'round':run['round'],'run_id':run['id']})
                for k,v in row.items():
                    if k.startswith('_'):continue
                    if k in batch:assert math.isclose(batch[k],v,abs_tol=1e-12)
                    batch[k]=v
    batches=[batches[s] for s in sorted(batches)]
    updates.sort(key=lambda r:r['update'])
    assert [b['step'] for b in batches]==list(range(1,81))
    assert [u['update'] for u in updates]==list(range(1,161))
    with (ROOT/'artifacts/analysis/training_steps.csv').open(encoding='utf-8') as f:
        local={int(r['step']):r for r in csv.DictReader(f)}
    checked=0;max_error=0.
    for row in batches:
        for key,value in local[row['step']].items():
            if key.startswith('rollout/') and value and key in row:
                error=abs(float(value)-row[key]);assert error<1e-12
                checked+=1;max_error=max(max_error,error)
    metrics=['train/entropy_loss','train/ppo_kl','train/pg_clipfrac','train/grad_norm','train/loss',
             'train/train_rollout_logprob_abs_diff','rollout/response_len/mean','rollout/truncated_ratio',
             'perf/step_time','perf/rollout_time','perf/actor_train_tok_per_s']
    rounds=[]
    for i in range(4):
        stats={}
        for k in metrics:
            vals=[r[k] for r in (updates if k.startswith('train/') else batches) if r['round']==i and k in r]
            stats[k]={'n':len(vals),'mean':statistics.mean(vals),'median':statistics.median(vals),'min':min(vals),'max':max(vals)} if vals else None
        rounds.append({'round':i,'metrics':stats,'zero_grad_updates':[u['update'] for u in updates if u['round']==i and u.get('train/grad_norm')==0]})
    csv_write('rollout_steps.csv',batches);csv_write('optimizer_updates.csv',updates)
    result={'runs':manifest['runs'],'steps':batches,'updates':updates,'rounds':rounds,
            'reconciliation':{'compared_numeric_values':checked,'max_abs_error':max_error},
            'step_mapping':'W&B train/step is 2..161; display update = train/step − 1; rollout step = floor(train/step / 2). W&B _step is a logging event index.',
            'scope':'Four completed rounds only, rollout steps 1..80, 160 optimizer updates'}
    write('summary.json',result)
    print('Verified',len(batches),'rollout steps,',len(updates),'updates;',checked,'local values; max difference',max_error)
    for r in rounds:
        print('Round',r['round'],{k:round(v['mean'],6) for k,v in r['metrics'].items() if v},'zero-grad updates',r['zero_grad_updates'])

if __name__=='__main__':main()
