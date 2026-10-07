"""Fixed-reference precision sensitivity on the first 32 validation documents."""
import argparse
import gc
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from .core import atomic_json, atomic_npz, digest
from .model import InstrumentedModel
from .precision_math import precise_rope
from .revised import context, metrics

NAMES = ["local_logit_gain_early_minus_late", "local_probability_gain_early_minus_late",
         "first_bias_kl_gain", "first_over_self_kl_gain"]


def endpoints(model, tokens, ref, cfg):
    layers = []
    def collect(layer, module, query, key, weights, scaling, mode):
        logits = model.distance_logits(module, query, key, scaling)
        loc, pre = metrics(logits, weights[0].cpu().numpy(), ref, layer, cfg)
        e0, e1 = cfg["early"]; l0, l1 = cfg["late"]; a, b = cfg["comparison_queries"]
        layers.append(np.stack([loc[0, :, e0:e1].mean(-1) - loc[0, :, l0:l1].mean(-1),
            loc[1, :, e0:e1].mean(-1) - loc[1, :, l0:l1].mean(-1),
            pre[3, :, a:b].mean(-1), pre[5, :, a:b].mean(-1)]).mean(-1))
    model.forward(tokens, collect)
    return np.stack(layers).mean(0)


def assess(native, precise):
    if not np.isfinite(native).all() or not np.isfinite(precise).all():
        raise ValueError("Non-finite endpoint precision comparison")
    delta = native - precise
    scale = np.maximum(np.sqrt(np.mean(precise**2, axis=0)), 1e-12)
    ratio = np.sqrt(np.mean(delta**2, axis=0)) / scale
    return {"passed": bool(np.all(ratio <= 0.01)), "relative_rms_error_limit": 0.01,
        "definition": "RMS paired endpoint difference / max(RMS float64 endpoint, 1e-12), separately for each endpoint",
        "endpoints": [{"name": name, "float32_mean": float(native[:, i].mean()),
            "float64_mean": float(precise[:, i].mean()), "mean_difference": float(delta[:, i].mean()),
            "max_absolute_document_difference": float(np.abs(delta[:, i]).max()),
            "relative_rms_error": float(ratio[i])} for i, name in enumerate(NAMES)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg, source, out, manifest, old, identity = context(parser.parse_args().config)
    with np.load(out / "reference.npz") as z:
        ref = {k: z[k].copy() for k in z.files if k != "identity"}
    ref_hash = digest({k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in ref.items()})
    audit_hash = digest(["precision-v3", identity, ref_hash, 32, .01])
    tokens = np.load(source / "tokens-validation.npz")["tokens"][:32]
    if len(tokens) != 32:
        raise ValueError("Precision protocol requires 32 validation documents")
    destination = out / "precision-sensitivity"
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 18 * 2**30:
        raise RuntimeError("Precision audit needs 18 GiB free GPU memory")
    results = []
    for kind in ("float32", "float64"):
        model = None
        values = []
        for i, document in enumerate(tokens):
            path = destination / kind / f"{i:05d}.npz"
            if path.exists():
                with np.load(path) as z:
                    if str(z["audit_hash"]) != audit_hash: raise ValueError("Stale precision checkpoint")
                    value = z["endpoints"].copy()
            else:
                if model is None:
                    model = InstrumentedModel(manifest["config"], manifest)
                    if kind == "float64": model.model.double()
                if kind == "float64":
                    with precise_rope(model.model, full=True):
                        value = endpoints(model, document, ref, cfg)
                else:
                    value = endpoints(model, document, ref, cfg)
                atomic_npz(path, endpoints=value, audit_hash=np.array(audit_hash),
                    document_id=np.array(manifest["splits"]["validation"][i]["id"]))
            values.append(value)
            print(f"Precision {kind}: {i+1}/32", flush=True)
        results.append(np.stack(values))
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    report = assess(*results)
    report.update(n_documents=32, length=int(tokens.shape[1]), audit_hash=audit_hash,
        scope="Evaluation sensitivity conditional on identical float32-fitted references; does not test refitting in float64",
        precision="float64 model, RoPE, RMSNorm, softmax and metrics; same stored checkpoint weights and frequencies",
        document_ids=[x["id"] for x in manifest["splits"]["validation"][:32]])
    atomic_json(destination / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Endpoint precision exceeds the fixed 1% RMS budget; test collection stopped for review")


if __name__ == "__main__":
    main()
