"""Retain strict sanity results and separately audit a numerical shift exception."""
import argparse
import gc
import json
from pathlib import Path
import torch
from .cli import load_config
from .core import atomic_json
from .experiment import sanity
from .precision_math import diagnose
from .model import InstrumentedModel
import numpy as np


def eligible(report):
    names = ("prefix_attention_max_error", "position_shift_attention_max_error",
             "repeated_token_identity_max_error", "tolerance")
    return (all(np.isfinite(report[n]) for n in names)
            and report["prefix_attention_max_error"] <= report["tolerance"]
            and report["repeated_token_identity_max_error"] <= report["tolerance"]
            and report["position_shift_attention_max_error"] > report["tolerance"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    source = Path(cfg["output"])
    try:
        sanity(cfg)
    except RuntimeError as error:
        if str(error) != "Sanity control failed; do not launch primary collection":
            raise
    report = json.loads((source / "sanity.json").read_text())
    if report["passed"]:
        atomic_json(source / "sanity-decision.json", {"decision": "strict checks passed"})
        return
    if not eligible(report):
        raise RuntimeError("A non-shift sanity check failed; no numerical exception is allowed")
    if cfg["model"] != "Qwen/Qwen2.5-1.5B" or cfg["dtype"] != "float32":
        raise RuntimeError("Numerical exception is restricted to float32 Qwen2.5-1.5B")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < 18 * 2**30:
            raise RuntimeError("Precision audit requires 18 GiB free GPU memory")
    manifest = json.loads((source / "manifest.json").read_text())
    observer = InstrumentedModel(cfg, manifest)
    rope = getattr(observer.model.config, "rope_scaling", None)
    if rope and rope.get("rope_type", rope.get("type", "default")) != "default":
        raise RuntimeError("Numerical exception requires fixed default RoPE")
    tokens = np.load(source / "tokens-validation.npz")["tokens"][0, :64]
    evidence = diagnose(observer.model, tokens)
    atomic_json(source / "precision-sanity.json", evidence)
    if not np.isfinite(evidence["float64_shift"]["max"]) or evidence["float64_shift"]["max"] > 1e-8:
        raise RuntimeError("Shift discrepancy persists in float64; do not collect")
    decision = {"decision": "proceed with documented numerical shift exception",
        "strict_sanity_passed": False, "original_tolerance": report["tolerance"],
        "float64_shift_max": evidence["float64_shift"]["max"], "float64_shift_limit": 1e-8,
        "scope": "64-token validation control only; endpoint precision audit required before test collection",
        "environment": observer.metadata}
    atomic_json(source / "sanity-decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
