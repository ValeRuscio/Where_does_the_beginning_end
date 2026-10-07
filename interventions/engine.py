"""Observe native float32 eager attention, without changing scores or masks."""
from contextlib import contextmanager
import importlib
import numpy as np
import torch
import transformers

class Engine:
    def __init__(self, model):
        if transformers.__version__ != '4.57.1':
            raise RuntimeError('Run setup_environment.py with this Python; transformers==4.57.1 is required')
        self.model = model.eval()
        self.device = next(model.parameters()).device
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError('This experiment uses float32 throughout')
        cfg = model.config
        if cfg.model_type not in {'qwen2','qwen3'} or cfg._attn_implementation != 'eager':
            raise ValueError('Only Qwen2/Qwen3 native eager attention is supported')
        if getattr(cfg,'use_sliding_window',False) or 'sliding_attention' in (getattr(cfg,'layer_types',None) or []):
            raise ValueError('This protocol requires full causal attention')
        rope = getattr(cfg,'rope_scaling',None)
        if rope and rope.get('rope_type',rope.get('type','default')) != 'default':
            raise ValueError('This protocol uses fixed standard RoPE')
        self.module = importlib.import_module(f'transformers.models.{cfg.model_type}.modeling_{cfg.model_type}')

    @contextmanager
    def observer(self, callback):
        from torch.overrides import TorchFunctionMode
        original = self.module.eager_attention_forward
        class Watch(TorchFunctionMode):
            def __init__(self):
                self.scores = []
            def __torch_function__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if func is torch.nn.functional.softmax:
                    self.scores.append(args[0] if args else kwargs['input'])
                return result
        def wrapper(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if query.shape[0] != 1 or query.shape[-2] != key.shape[-2]:
                raise RuntimeError('Batch size one, complete prefix, no cache required')
            if attention_mask is None:
                raise RuntimeError('Expected native causal mask')
            with Watch() as seen:
                result = original(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
            if len(seen.scores) != 1 or result[1] is None:
                raise RuntimeError('Native attention capture failed')
            callback(int(module.layer_idx),seen.scores[0][0],result[1][0])
            return result
        self.module.eager_attention_forward = wrapper
        try:
            yield
        finally:
            self.module.eager_attention_forward = original

    def collect(self, tokens, suffix_start=0, suffix_length=512, reference=None, retain_full=False, audit=False, position_shift=0):
        if len(tokens) != suffix_start+suffix_length or suffix_length < 8:
            raise ValueError('Expected prefix followed by exactly the measured suffix')
        q = torch.arange(suffix_start+4,len(tokens),device=self.device)
        keys = q[:,None]-torch.arange(4,device=self.device)[None,:]
        pieces = {k:[] for k in ['z','u','first','suffix','full_tv']}
        full, order, errors = [], [], []
        def capture(layer,scores,weights):
            order.append(layer)
            # Use the actual native softmax input, not a sliced QK recomputation.
            local = scores[:,q[:,None],keys]
            centered = local-local.mean(-1,keepdim=True)
            probs = torch.softmax(local,dim=-1)
            arrays = {'z':centered,'u':probs,'first':weights[:,q,0],'suffix':weights[:,q,suffix_start]}
            for k,a in arrays.items():
                if not bool(torch.isfinite(a).all()):
                    raise RuntimeError(f'Non-finite native measurement at layer {layer}: {k}')
                pieces[k].append(a.cpu().numpy().copy())
            if reference is not None:
                base = torch.as_tensor(reference[layer],device=self.device)
                if base.shape != weights.shape:
                    raise ValueError('Full attention comparison requires identical support')
                delta = .5*(weights[:,q,:]-base[:,q,:]).abs().sum(-1)
                if not bool(torch.isfinite(delta).all()):
                    raise RuntimeError('Non-finite full-distribution response')
                pieces['full_tv'].append(delta.cpu().numpy().copy())
            if retain_full:
                full.append(weights.cpu().numpy().copy())
            if audit:
                expected = torch.softmax(scores,dim=-1,dtype=torch.float32)
                error = float((expected-weights).abs().max())
                if not bool(torch.isfinite(weights).all()) or error > 2e-6:
                    raise RuntimeError(f'Native capture mismatch: {error}')
                if float(weights.triu(1).abs().max()) != 0:
                    raise RuntimeError('Noncausal attention detected')
                if float((weights.sum(-1)-1).abs().max()) > 2e-5:
                    raise RuntimeError('Attention is not normalized')
                errors.append(error)
        ids = torch.as_tensor(np.asarray(tokens).copy(),dtype=torch.long,device=self.device)[None]
        positions = torch.arange(len(tokens),device=self.device)[None]+position_shift
        with torch.inference_mode(), self.observer(capture):
            output = self.model(input_ids=ids,position_ids=positions,use_cache=False,output_attentions=False)
            last = output.last_hidden_state.detach().cpu().numpy().copy() if audit else None
            del output
        if order != list(range(self.model.config.num_hidden_layers)):
            raise RuntimeError(f'Unexpected capture order {order}')
        result = {k:np.stack(v) for k,v in pieces.items() if v}
        if retain_full:
            result['full'] = full
        if audit:
            result['capture_max_error'] = max(errors)
            result['hidden'] = last
        return result

    def preflight(self,tokens):
        # Same-size repeat and future-token mutation isolate implementation errors
        # without invoking the previously diagnosed position-ID shift exception.
        short = np.asarray(tokens[:64]).copy()
        a = self.collect(short,suffix_length=64,audit=True)
        b = self.collect(short,suffix_length=64,audit=True)
        repeat = max(float(np.max(np.abs(a[k]-b[k]))) for k in ['z','u','first','hidden'])
        changed = short.copy()
        changed[-1] = (int(changed[-1])+1) % self.model.config.vocab_size
        c = self.collect(changed,suffix_length=64,audit=True)
        causal = max(float(np.max(np.abs(a[k][...,:-1,:]-c[k][...,:-1,:]))) for k in ['z','u'])
        ids = torch.as_tensor(short,dtype=torch.long,device=self.device)[None]
        with torch.inference_mode():
            native = self.model(input_ids=ids,position_ids=torch.arange(64,device=self.device)[None],use_cache=False).last_hidden_state.cpu().numpy()
        observer_error = float(np.max(np.abs(native-a['hidden'])))
        if max(repeat,causal,observer_error) > 2e-5:
            raise RuntimeError(f'Preflight failed: repeat={repeat}, causal={causal}, observer={observer_error}')
        return {'passed':True,'repeat_max_error':repeat,'future_token_control_max_error':causal,
                'observation_hidden_max_error':observer_error,'capture_max_error':a['capture_max_error'],
                'note':'Native float32 checks; not a float64 intervention-effect audit.'}
