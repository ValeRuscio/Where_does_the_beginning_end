#!/usr/bin/env python3
"""Standalone runner: no editable install, old package, or reference fit needed."""
import argparse
import gc
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
from design import MODELS,sha,digest,validate,load_inputs

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results'/'.matplotlib'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)

def atomic_npz(path,value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,**value)
    temporary.replace(path)

def initialize(cfg,stage,output,model):
    split = 'validation' if stage=='pilot' else cfg['split']
    count = cfg['pilot_documents'] if stage=='pilot' else cfg['n_documents']
    tokens,ids = load_inputs(ROOT,split)
    if count > len(tokens):
        raise ValueError('Not enough documents')
    tokens,ids = tokens[:count],ids[:count]
    sources = {p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}
    payload = {'config':cfg,'stage':stage,'model':model,'checkpoint':MODELS[model],
               'split':split,'document_ids':ids,'token_sha256':sha(ROOT/'input_snapshot'/f'tokens-{split}.npz'),
               'donor_input_sha256':sha(ROOT/'input_snapshot'/'tokens-train.npz'),
               'source_hashes':sources,'protocol_version':'boundary-interventions-1'}
    identity = digest(payload)
    directory = output/stage/model;directory.mkdir(parents=True,exist_ok=True)
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['identity'] != identity:
            raise RuntimeError(f'Config/code/input changed for {directory}. Use a NEW --output directory; old results are preserved.')
    else:
        atomic_json(manifest_path,{**payload,'identity':identity})
    (directory/'documents').mkdir(exist_ok=True)
    return directory,tokens,ids,identity

def run_model(cfg,stage,output,name):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from engine import Engine
    from experiment import document
    directory,tokens,ids,identity = initialize(cfg,stage,output,name)
    pending = []
    for i,doc in enumerate(ids):
        path = directory/'documents'/f'{i:04d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['document_id'].item()) != doc or str(z['run_identity'].item()) != identity:
                    raise RuntimeError(f'Wrong result identity: {path}')
                if not np.isfinite(z['endpoints']).all():
                    raise RuntimeError(f'Invalid checkpoint: {path}')
        else:
            pending.append(i)
    if not pending:
        print(f'{name}: all {len(ids)} documents already complete',flush=True);return
    if not torch.cuda.is_available():
        raise RuntimeError('This runner needs the CUDA Python/kernel. No CPU fallback for full collection.')
    free,total = torch.cuda.mem_get_info()
    if free < 12*1024**3:
        raise RuntimeError(f'Only {free/1024**3:.1f} GiB GPU memory free. Stop other GPU jobs or restart old model-holding kernels. Need >=12 GiB free.')
    if transformers.__version__ != '4.57.1':
        raise RuntimeError('Run setup_environment.py with this same Python first')
    torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model_id,revision = MODELS[name]
    print(f'Loading {model_id} at {revision} in float32; {free/1024**3:.1f} GiB free',flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id,revision=revision,dtype=torch.float32,
                                     attn_implementation='eager',trust_remote_code=False).to('cuda').eval()
    engine = Engine(model)
    train,_ = load_inputs(ROOT,'train')
    if int(max(tokens.max(),train.max())) >= model.config.vocab_size or int(min(tokens.min(),train.min())) < 0:
        raise RuntimeError('Snapshot token ID outside model vocabulary')
    environment = {'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,
                   'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'dtype':'float32',
                   'model_config':model.config.to_dict()}
    prior_env = directory/'environment.json'
    if prior_env.exists():
        prior = json.loads(prior_env.read_text())
        for k in ['torch','transformers','numpy','python','gpu','dtype']:
            if prior[k] != environment[k]:
                raise RuntimeError(f'Resume environment changed ({k}); use a new output directory')
    atomic_json(prior_env,environment)
    audit = engine.preflight(tokens[pending[0],256:768])
    atomic_json(directory/'preflight.json',audit)
    print(f'{name}: native capture, unchanged-input repeat, and future-token checks passed',flush=True)
    started = time.monotonic()
    for j,i in enumerate(pending):
        t = time.monotonic()
        result = document(engine,tokens[i],cfg,ids[i],train,tokenizer.all_special_ids,audit=(j==0))
        result['run_identity'] = np.asarray(identity)
        atomic_npz(directory/'documents'/f'{i:04d}.npz',result)
        elapsed = time.monotonic()-started
        remaining = elapsed/(j+1)*(len(pending)-j-1)
        print(f'{name} {i+1}/{len(ids)} | source {ids[i]} | {time.monotonic()-t:.1f}s | estimated remaining for THIS model {remaining/60:.1f} min | endpoints {result["endpoints"].tolist()}',flush=True)
        atomic_json(directory/'progress.json',{'completed':len(ids)-len(pending)+j+1,'total':len(ids),
                                            'remaining_estimate_minutes':remaining/60,'last_document':ids[i]})
    del engine,model,tokenizer;gc.collect();torch.cuda.empty_cache()

def package(output,stage):
    target = output/f'{stage}-intervention-results.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((output/stage).rglob('*')):
            if p.is_file() and p.suffix != '.tmp':
                z.write(p,p.relative_to(output))
        for p in sorted(ROOT.glob('*.py')):
            z.write(p,Path('code')/p.name)
        for filename in ['config.json','PROTOCOL.txt','requirements.txt','bundle-integrity.json']:
            z.write(ROOT/filename,Path('code')/filename)
    print(f'Results archive: {target}',flush=True)
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','analyze','package','verify'])
    parser.add_argument('--stage',choices=['pilot','main'],default='pilot')
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--output',default=str(ROOT/'results'))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text());validate(cfg)
    output = Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    expected = json.loads((ROOT/'bundle-integrity.json').read_text())
    for relative,checksum in expected['inputs'].items():
        if sha(ROOT/relative) != checksum:
            raise RuntimeError(f'Input snapshot mismatch: {relative}')
    if args.command == 'verify':
        for split in ['train','validation','test']:load_inputs(ROOT,split)
        print('All bundled inputs and source IDs verified. No old run folder is required.');return
    if args.command == 'package':package(output,args.stage);return
    from report import analyze
    if args.command == 'analyze':
        analyze(ROOT,cfg,args.stage,output/args.stage);return
    if args.stage == 'main':
        for name in cfg['models']:
            pilot_dir,_,_,_ = initialize(cfg,'pilot',output,name)
            summary_path=pilot_dir/'summary.json'
            if not summary_path.exists() or not json.loads(summary_path.read_text()).get('complete'):
                raise RuntimeError('Run the pilot with this config/output first: python run_experiments.py run --stage pilot')
    import fcntl
    lock_path = output/'.collection.lock'
    with lock_path.open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another collection process is already using this output folder')
        atomic_json(output/'job-status.json',{'status':'running','pid':os.getpid(),'stage':args.stage})
        try:
            for model in cfg['models']:run_model(cfg,args.stage,output,model)
            result = analyze(ROOT,cfg,args.stage,output/args.stage)
            if not result['complete']:raise RuntimeError('Collection ended with missing document results')
            package(output,args.stage)
            atomic_json(output/'job-status.json',{'status':'complete','stage':args.stage})
        except BaseException as e:
            atomic_json(output/'job-status.json',{'status':'failed','stage':args.stage,'error':repr(e)})
            raise

if __name__ == '__main__':main()
