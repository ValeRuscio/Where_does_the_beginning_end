"""Read-only diagnostic: no training, no test split, no changes to experiment code."""
from pathlib import Path
import contextlib
import json
import types
import numpy as np
import torch
import transformers
from torch.overrides import TorchFunctionMode


def summary(a, b):
    delta = np.abs(a - b)
    index = np.unravel_index(delta.argmax(), delta.shape)
    return {"max": float(delta.max()), "mean": float(delta.mean()),
            "p99": float(np.quantile(delta, .99)),
            "worst_layer_head_query_key": [int(x) for x in index],
            "max_by_layer": delta.max(axis=(1, 2, 3)).tolist()}


@contextlib.contextmanager
def precise_rope(model, full=False):
    """Diagnostic only: same stored frequencies, evaluated in float64."""
    changed = []
    def rope(self, x, position_ids):
        frequency = self.inv_freq.double().to(x.device)
        angles = position_ids.double()[..., None] * frequency[None, None, :]
        angles = torch.cat((angles, angles), dim=-1)
        return ((angles.cos() * self.attention_scaling).to(x.dtype),
                (angles.sin() * self.attention_scaling).to(x.dtype))
    def norm(self, hidden_states):
        variance = hidden_states.square().mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    class PreciseSoftmax(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func is torch.nn.functional.softmax and full:
                kwargs["dtype"] = torch.float64
            return func(*args, **kwargs)
    try:
        for module in model.modules():
            fn = None
            if type(module).__name__ == "Qwen2RotaryEmbedding": fn = rope
            if full and type(module).__name__ == "Qwen2RMSNorm": fn = norm
            if fn:
                changed.append((module, module.forward))
                module.forward = types.MethodType(fn, module)
        if not changed:
            raise RuntimeError("Expected Qwen2 rotary module was not found")
        with PreciseSoftmax():
            yield
    finally:
        for module, forward in changed:
            module.forward = forward


def collect(model, tokens, shift=0):
    device = next(model.parameters()).device
    ids = torch.tensor(tokens, dtype=torch.long, device=device)[None]
    positions = torch.arange(len(tokens), device=device)[None] + shift
    with torch.inference_mode():
        output = model(input_ids=ids, position_ids=positions,
                       use_cache=False, output_attentions=True)
        if not output.attentions or any(x is None for x in output.attentions):
            raise RuntimeError("Native eager attention was not returned")
        result = np.stack([x[0].double().cpu().numpy() for x in output.attentions])
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite attention in diagnostic")
    return result


def diagnose(model, tokens):
    report = {}
    base = collect(model, tokens)
    report["native_float32_shift"] = summary(base, collect(model, tokens, 37))
    report["native_float32_repeat_same_input"] = summary(base, collect(model, tokens))
    print("Native float32 shift:", report["native_float32_shift"]["max"], flush=True)
    with precise_rope(model):
        rope_base = collect(model, tokens)
        report["float32_with_float64_trigonometry_shift"] = summary(rope_base, collect(model, tokens, 37))
        report["native_vs_precise_trigonometry_at_position_zero"] = summary(base, rope_base)
    print("Float64 trigonometry shift:", report["float32_with_float64_trigonometry_shift"]["max"], flush=True)
    model.double()
    with precise_rope(model, full=True):
        reference = collect(model, tokens)
        report["float64_shift"] = summary(reference, collect(model, tokens, 37))
        report["native_float32_vs_float64_at_position_zero"] = summary(base, reference)
    print("Float64 shift:", report["float64_shift"]["max"], flush=True)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert transformers.__version__ == "4.57.1", "Use the existing pinned experiment Python"
    assert torch.cuda.is_available(), "Use the working GPU Python"
    root = args.root.resolve()
    cfg = json.loads((root / "configs/replication-qwen25.json").read_text())
    source = root / cfg["output"]
    manifest = json.loads((source / "manifest.json").read_text())
    assert cfg["dtype"] == "float32", "This diagnostic expects the float32 experiment"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    free, _ = torch.cuda.mem_get_info()
    if free < 18 * 2**30:
        raise RuntimeError("This diagnostic needs 18 GiB free for a float64 Qwen2.5-1.5B model; stop your own unused GPU jobs first")
    model = transformers.AutoModel.from_pretrained(cfg["model"], revision=manifest["model_revision"],
        dtype=torch.float32, attn_implementation="eager", trust_remote_code=False).to("cuda").eval()
    assert model.config.model_type == "qwen2"
    rope = getattr(model.config, "rope_scaling", None)
    assert not rope or rope.get("rope_type", rope.get("type", "default")) == "default", "Diagnostic requires fixed default RoPE"
    tokens = np.load(source / "tokens-validation.npz")["tokens"][0, :64]
    report = diagnose(model, tokens)
    report["model"] = cfg["model"]
    report["revision"] = manifest["model_revision"]
    report["gpu"] = torch.cuda.get_device_name(0)
    report["torch"] = torch.__version__
    report["note"] = "Diagnostic only; same first validation passage, 64 tokens, shift 37. No training, test access or tolerance changes."
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
