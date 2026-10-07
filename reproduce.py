#!/usr/bin/env python3
"""Repository entry point. Run from any working directory; no package install needed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

def command(args,cwd):
    cwd=Path(cwd);(cwd/'logs').mkdir(exist_ok=True)
    env=os.environ.copy();env['PYTHONPATH']=str(cwd)
    env['TOKENIZERS_PARALLELISM']='false';env['MPLBACKEND']='Agg'
    env['MPLCONFIGDIR']=str(ROOT/'generated'/'.matplotlib')
    print('Running:', ' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,'-u',*map(str,args)],cwd=cwd,env=env,check=True)

def verify():
    manifest=json.loads((ROOT/'CHECKSUMS.json').read_text())
    for name,expected in manifest.items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Missing/modified distributed file: {name}')
    print(f'All {len(manifest)} distributed files verified.')

def predictive(model,stage,split):
    cwd=ROOT/'predictive'/model
    def run(*args):command(args,cwd)
    if stage in ('all','prepare'):
        run('-m','boundary_attention.replication','prepare','--config','base.json','--snapshot','../../data/input_snapshot')
    if stage in ('all','fit'):
        if model=='qwen25':run('-m','boundary_attention.audited_sanity','--config','base.json')
        else:run('-m','boundary_attention.cli','sanity','--config','base.json')
        run('-m','boundary_attention.gpu_preflight','--config','base.json')
        run('-m','boundary_attention.cli','fit','--config','base.json')
        run('-m','boundary_attention.revised','fit','--config','revised.json')
        if model=='qwen25':run('-m','boundary_attention.precision_endpoints','--config','revised.json')
    allow=['--allow-test'] if split=='test' else []
    if stage in ('all','evaluate'):
        if model=='qwen25':
            cfg=json.loads((cwd/'revised.json').read_text())
            audit=cwd/cfg['output']/'precision-sensitivity'/'summary.json'
            if not audit.exists() or not json.loads(audit.read_text()).get('passed'):
                raise RuntimeError('Run the Qwen2.5 fitting/precision stage before evaluation')
        run('-m','boundary_attention.revised','evaluate','--config','revised.json','--split',split,*allow)
    if stage in ('all','analyze'):
        run('-m','boundary_attention.revised','analyze','--config','revised.json','--split',split,*allow)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='task',required=True)
    for name in ['verify','analyze','figures','test']:sub.add_parser(name)
    pred=sub.add_parser('predictive');pred.add_argument('--model',choices=['qwen3','qwen25'],required=True)
    pred.add_argument('--stage',choices=['all','prepare','fit','evaluate','analyze'],default='all')
    pred.add_argument('--split',choices=['test','validation'],default='test')
    inter=sub.add_parser('interventions');inter.add_argument('--stage',choices=['pilot','main'],default='pilot')
    a=p.parse_args()
    if a.task=='verify':verify()
    elif a.task=='analyze':command(['analyze_saved.py'],ROOT)
    elif a.task=='figures':
        for file in ['build_figures.py','build_intervention_figures.py']:command([file],ROOT/'paper_assets')
    elif a.task=='test':
        for cwd in [ROOT/'predictive/qwen3',ROOT/'predictive/qwen25',ROOT/'interventions']:
            command(['-m','unittest','discover','-s','tests','-v'],cwd)
    elif a.task=='predictive':predictive(a.model,a.stage,a.split)
    else:command(['run_experiments.py','run','--stage',a.stage],ROOT/'interventions')

if __name__=='__main__':main()
