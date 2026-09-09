#!/usr/bin/env python3
"""Read the four completed experiment runs; export only numeric history and public metadata.

WANDB_API_KEY is read in memory from the environment or the project .env.
No login(), run creation, remote edits, console logs, configs, or media exports.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / 'artifacts/analysis/wandb'
RUNS=['8ov14yyc','chjizc8u','7vhbv5j0','y4xk6byf']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-path',type=Path)
    args=parser.parse_args()
    if args.sdk_path: sys.path.insert(0,str(args.sdk_path.resolve()))
    os.environ['WANDB_SILENT']='true'
    key=os.environ.get('WANDB_API_KEY')
    if not key:
        for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines():
            name,sep,value=line.strip().removeprefix('export ').partition('=')
            if sep and name.strip()=='WANDB_API_KEY':
                key=value.strip().strip('\"\''); break
    if not key: raise SystemExit('WANDB_API_KEY is missing')
    import wandb
    api=wandb.Api(api_key=key,timeout=45)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={'fetched_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'method':'Run.scan_history() without key filtering or sampling',
              'sdk_version':wandb.__version__,'runs':[]}
    for i,rid in enumerate(RUNS):
        run=api.run('hzj-lab/shopsimrl/'+rid)
        rows=[]
        for row in run.scan_history(page_size=1000):
            numeric={k:v for k,v in row.items() if isinstance(v,(int,float)) and math.isfinite(v)}
            rows.append(numeric)
        metadata={'id':rid,'url':f'https://wandb.ai/hzj-lab/shopsimrl/runs/{rid}',
                  'round':i if i<4 else None,'primary':i<4,'state':run.state,
                  'created_at':run.created_at,'history_rows':len(rows),
                  'keys':sorted({k for r in rows for k in r})}
        path=OUT / (rid+'.json')
        path.write_text(json.dumps({'metadata':metadata,'history':rows},indent=2,allow_nan=False)+'\n',encoding='utf-8')
        manifest['runs'].append({**metadata,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(rid,run.state,'rows',len(rows),'numeric keys',len(metadata['keys']),flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        # Do not print request objects or an SDK traceback that could contain credentials.
        raise SystemExit('W&B read failed: '+type(exc).__name__)
