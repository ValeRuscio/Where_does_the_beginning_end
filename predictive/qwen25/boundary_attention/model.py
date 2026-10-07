"""Capture the tensors actually used by supported eager attention implementations.

This observes q/k AFTER RoPE and QK normalization. No projection reconstruction,
no probes, no changes to native causal attention during the main experiment.
"""
from contextlib import contextmanager
import importlib
import platform
import numpy as np

SUPPORTED = {"qwen3", "qwen2", "llama", "mistral"}


class InstrumentedModel:
    def __init__(self, config, manifest):
        import torch
        import transformers
        from transformers import AutoModel
        if transformers.__version__ != "4.57.1":
            raise RuntimeError("Instrumentation is audited for transformers==4.57.1; use the pinned environment")
        self.torch = torch
        self.config = config
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
        device = config["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and dtype != torch.float32:
            raise ValueError("For CPU, explicitly select float32 (GPU recommended for the full study)")
        if dtype == torch.bfloat16 and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bfloat16; use float32 or documented float16 sensitivity run")
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = AutoModel.from_pretrained(config["model"], revision=manifest["model_revision"],
            torch_dtype=dtype, attn_implementation="eager", trust_remote_code=False).to(device).eval()
        self.device = device
        mc = self.model.config
        if mc.model_type not in SUPPORTED:
            raise ValueError(f"Unsupported model_type={mc.model_type}; do not silently use a different adapter")
        # The full-prefix null and diagonal extraction assume native full causal attention.
        # A native window needs a different support-aware null; fail instead of mislabelling it.
        layers = getattr(mc, "layer_types", None)
        if getattr(mc, "use_sliding_window", False) or (layers and "sliding_attention" in layers):
            raise ValueError("Native sliding attention not supported by the primary full-prefix null")
        if mc.model_type == "mistral" and getattr(mc, "sliding_window", None) is not None:
            raise ValueError("Use Mistral-7B-v0.3 with sliding_window=null for this protocol")
        rope = getattr(mc, "rope_scaling", None)
        if rope and rope.get("rope_type", rope.get("type", "default")) in {"dynamic", "longrope"}:
            raise ValueError("Length-dependent RoPE requires separate handling of prefix invariance")
        self.module = importlib.import_module(f"transformers.models.{mc.model_type}.modeling_{mc.model_type}")
        if not hasattr(self.module, "eager_attention_forward"):
            raise RuntimeError("Expected eager implementation not found")
        self.metadata = {"model_config": mc.to_dict(), "torch": torch.__version__,
            "capture_version": "native-softmax-input-v2",
            "transformers": transformers.__version__, "python": platform.python_version(),
            "device": device, "dtype": config["dtype"], "model_revision": manifest["model_revision"],
            "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None}

    @contextmanager
    def capture(self, callback, mode="causal"):
        torch = self.torch
        from torch.overrides import TorchFunctionMode
        original = self.module.eager_attention_forward
        calls = []
        class ObserveSoftmax(TorchFunctionMode):
            def __init__(self):
                self.inputs = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if func is torch.nn.functional.softmax:
                    self.inputs.append(args[0] if args else kwargs["input"])
                return result

        def wrapper(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if query.shape[0] != 1 or query.shape[-2] != key.shape[-2]:
                raise RuntimeError("Protocol requires batch=1, no KV cache, and a complete prefix")
            supplied_mask = attention_mask
            if mode == "bidirectional":
                supplied_mask = None
            elif mode != "causal":
                raise ValueError(mode)
            if supplied_mask is not None and bool((supplied_mask.tril() != 0).any()):
                raise RuntimeError("Expected zero additive mask on every accessible key")
            with ObserveSoftmax() as observed:
                result = original(module, query, key, value, supplied_mask, scaling=scaling,
                                  dropout=dropout, **kwargs)
            if len(observed.inputs) != 1:
                raise RuntimeError(f"Expected one native softmax, observed {len(observed.inputs)}")
            weights = result[1]
            if weights is None:
                raise RuntimeError("Eager implementation did not return weights")
            layer = int(module.layer_idx)
            calls.append(layer)
            self._native_capture = (module, query, key, observed.inputs[0], weights)
            try:
                callback(layer, module, query, key, weights, float(scaling), mode)
            finally:
                del self._native_capture
            return result
        self.module.eager_attention_forward = wrapper
        try:
            yield calls
        finally:
            self.module.eager_attention_forward = original

    def forward(self, tokens, callback, mode="causal", shift=0):
        torch = self.torch
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device)[None, :]
        positions = torch.arange(len(tokens), device=self.device)[None, :] + shift
        with torch.inference_mode(), self.capture(callback, mode) as calls:
            output = self.model(input_ids=ids, position_ids=positions,
                                use_cache=False, output_attentions=False)
            del output
        if calls != list(range(self.model.config.num_hidden_layers)):
            raise RuntimeError(f"Expected one capture per layer in order; observed {calls}")

    def distance_logits(self, module, query, key, scaling):
        """Float32 scores on accessible keys, converted to relative distance.

        Future entries are NaN. A separate original-softmax check catches adapter errors.
        """
        torch = self.torch
        captured = getattr(self, "_native_capture", None)
        if captured is not None and query.dtype in (torch.float32, torch.float64):
            if captured[0] is not module or captured[1] is not query or captured[2] is not key:
                raise RuntimeError("Native logit capture does not match this layer")
            # These are the actual scores entering native softmax. On allowed
            # keys the standard full causal mask adds zero. Future keys are
            # discarded below; no second QK multiplication is necessary.
            logits = captured[3][0]
        else:
            expanded_key = key.repeat_interleave(module.num_key_value_groups, dim=1)
            logits = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2))[0] * scaling
        length = logits.shape[-1]
        q = torch.arange(length, device=logits.device)[:, None]
        d = torch.arange(length, device=logits.device)[None, :]
        keys = (q - d).clamp(min=0).expand(logits.shape[0], -1, -1)
        by_distance = logits.gather(-1, keys).masked_fill((d > q)[None], float("nan"))
        return by_distance.cpu().numpy()

    def check_weights(self, query, key, module, weights, scaling, mode):
        """Check weights against the actual input to the native softmax.

        Slicing query rows before matmul can select a different CUDA kernel and
        introduce rounding differences unrelated to whether capture is correct.
        """
        torch = self.torch
        length = query.shape[-2]
        captured = getattr(self, "_native_capture", None)
        if captured is not None:
            if captured[0] is not module or captured[1] is not query or captured[2] is not key or captured[4] is not weights:
                raise RuntimeError("Native weight capture does not match this layer")
            logits = captured[3]
        else:
            keys = self.module.repeat_kv(key, module.num_key_value_groups)
            logits = torch.matmul(query, keys.transpose(2, 3)) * scaling
            if mode == "causal":
                positions = torch.arange(length, device=query.device)
                mask = positions[None, :] > positions[:, None]
                logits = logits.masked_fill(mask[None, None], float("-inf"))
        expected = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        actual = weights
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise RuntimeError("Non-finite attention weights in capture check")
        tolerance = 0.025 if query.dtype != torch.float32 else 2e-5
        error = float((expected.float() - actual.float()).abs().max())
        if error > tolerance:
            raise RuntimeError(f"Attention capture mismatch: {error}; inspect model adapter")
        if mode == "causal" and float(weights.triu(1).abs().max()) != 0:
            raise RuntimeError("Main run includes attention to future positions")
        return error
