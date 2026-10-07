"""Pure NumPy design and estimands. No model imports or GPU required."""
import hashlib
import json
from pathlib import Path
import numpy as np

MODELS = {
    'qwen25': ('Qwen/Qwen2.5-1.5B', '8faed761d45a263340a0528343f099c05c9a4323'),
    'qwen3': ('Qwen/Qwen3-1.7B-Base', 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def rng_for(seed, document, purpose):
    key = f'{seed}:{document}:{purpose}'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))

def validate(cfg):
    # Fix supports and estimands across runs rather than silently accepting a
    # config edit that invalidates index arithmetic or the registered endpoints.
    expected = {'suffix_start':256, 'suffix_length':512, 'prefix_lengths':[0,16,64,256],
                'shuffle_prefix_length':64, 'replacement_positions':[0,1]}
    for k,v in expected.items():
        if cfg.get(k) != v:
            raise ValueError(f'{k} must be {v!r} for this protocol')
    if not cfg['models'] or len(set(cfg['models'])) != len(cfg['models']) or any(k not in MODELS for k in cfg['models']):
        raise ValueError('Select qwen25 and/or qwen3 exactly once')
    if cfg['split'] not in {'test','validation'} or not 2 <= cfg['n_documents'] <= (512 if cfg['split']=='test' else 128):
        raise ValueError('Invalid evaluation split/document count')
    if not 1 <= cfg['pilot_documents'] <= 32 or not 1 <= cfg['replacement_donors'] <= 16:
        raise ValueError('Invalid pilot/donor count')
    if cfg['bootstrap_replicates'] < 100:
        raise ValueError('At least 100 bootstrap replicates required')

def load_inputs(root, split):
    folder = Path(root)/'input_snapshot'
    manifest = json.loads((folder/'manifest.json').read_text())
    with np.load(folder/f'tokens-{split}.npz', allow_pickle=False) as z:
        tokens = z['tokens'].copy()
    ids = [str(x['id']) for x in manifest['splits'][split]]
    if tokens.shape != (len(ids),1024) or tokens.dtype.kind not in 'iu':
        raise ValueError('Token snapshot has incorrect shape or dtype')
    all_ids = [str(x['id']) for rows in manifest['splits'].values() for x in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Source IDs overlap within/between splits')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source IDs')
    return tokens, ids

def boundary_conditions(tokens, cfg, doc_id):
    start, size = cfg['suffix_start'], cfg['suffix_length']
    suffix = tokens[start:start+size]
    items = []
    for m in cfg['prefix_lengths']:
        sequence = tokens[start-m:start+size].copy()
        assert np.array_equal(sequence[m:], suffix)
        items.append((f'prefix_{m}', sequence, m))
    m = cfg['shuffle_prefix_length']
    sequence = tokens[start-m:start+size].copy()
    interior = sequence[1:m-1].copy()
    order = rng_for(cfg['seed'], doc_id, 'shuffle').permutation(len(interior))
    sequence[1:m-1] = interior[order]
    assert np.array_equal(np.sort(sequence[:m]), np.sort(tokens[start-m:start]))
    assert sequence[0] == tokens[start-m] and sequence[m-1] == tokens[start-1]
    assert np.array_equal(sequence[m:], suffix)
    items.append(('shuffle_64', sequence, m))
    return items

def donor_tokens(train, original, specials, count, seed, doc_id):
    # Uniform over unique ordinary token identities at the same crop coordinate
    # in fitting documents; fixed donor pool, conditional inference.
    pool = np.unique(train[:,256])
    excluded = set(map(int,specials)) | {int(original[0]),int(original[1])}
    pool = np.asarray([t for t in pool if int(t) not in excluded], dtype=np.int64)
    if len(pool) < count:
        raise ValueError('Too few distinct eligible donor identities')
    return rng_for(seed, doc_id, 'donors').choice(pool,count,replace=False)

def tv(p,q):
    return .5*np.abs(np.asarray(p,dtype=np.float64)-np.asarray(q,dtype=np.float64)).sum(axis=-1)

def band(curve, low, high):
    # Every saved query axis corresponds to suffix positions 4..511 inclusive.
    return np.asarray(curve)[...,low-4:high-4+1].mean(axis=-1)

def primary_head_endpoints(boundary_tv, replacement_tv):
    # Input shape: boundary [condition,layer,head,query]; replacement [position,layer,head,query].
    a = band(boundary_tv[2],4,7)-band(boundary_tv[2],32,63)
    b = band(replacement_tv[0],5,63)-band(replacement_tv[1],5,63)
    return np.stack([a,b])
