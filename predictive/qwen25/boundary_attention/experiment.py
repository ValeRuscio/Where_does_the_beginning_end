import json
import time
from pathlib import Path
import numpy as np
from .core import atomic_json, atomic_npz, digest, softmax, relative_null, js_divergence
from .model import InstrumentedModel

METRICS = ["logit_correction_gain", "conditional_probability_correction_gain",
           "prefix_null_tv", "prefix_null_js", "prefix_uniform_tv",
           "first_key_mass", "effective_keys", "entropy", "future_mass",
           "matched_offset_log_ratio_error", "logit_baseline_mse", "conditional_baseline_mse"]


def read_run(config):
    out = Path(config["output"])
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["config_digest"] != digest(config):
        raise ValueError("Config differs from prepared manifest; use a new output directory")
    return out, manifest


def training_features(logits, fit_query_min, common_offsets):
    """logits is [head,query,distance], with NaN outside causal support."""
    centered = logits - logits[..., :1]
    tail = centered[:, fit_query_min:, :]
    count = np.isfinite(tail).sum(axis=1)
    if (count == 0).any():
        raise ValueError("Reference profile has an unobserved offset")
    profile = np.nansum(tail, axis=1, dtype=np.float64) / count
    common = centered[..., :common_offsets].copy()
    common[:, :common_offsets-1] = 0
    common -= common.mean(axis=-1, keepdims=True)
    probabilities = softmax(common)
    return profile, common, probabilities


def fit(config):
    out, manifest = read_run(config)
    tokens = np.load(out / "tokens-train.npz")["tokens"]
    model = InstrumentedModel(config, manifest)
    atomic_json(out / "environment.json", model.metadata)
    checkpoint = out / "fit-checkpoint.npz"
    start, sums = 0, None
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            if str(z["manifest_digest"]) != digest(manifest):
                raise ValueError("Stale checkpoint")
            start = int(z["completed"])
            sums = [z[name] for name in ["profile_sum", "common_logit_sum", "common_probability_sum"]]
    for index in range(start, len(tokens)):
        layers = []
        def collect(layer, module, query, key, weights, scaling, mode):
            if index == start:
                model.check_weights(query, key, module, weights, scaling, mode)
            logits = model.distance_logits(module, query, key, scaling)
            layers.append(training_features(logits, config["fit_query_min"], config["common_offsets"]))
        begin = time.perf_counter()
        model.forward(tokens[index], collect)
        arrays = [np.stack([x[j] for x in layers]).astype(np.float64) for j in range(3)]
        if sums is None:
            sums = arrays
        else:
            for total, value in zip(sums, arrays):
                total += value
        atomic_npz(checkpoint, completed=np.array(index + 1), manifest_digest=np.array(digest(manifest)),
                   profile_sum=sums[0], common_logit_sum=sums[1], common_probability_sum=sums[2])
        print(f"Fit {index+1}/{len(tokens)}: {time.perf_counter()-begin:.1f}s", flush=True)
    if sums is None:
        raise ValueError("Empty training split")
    atomic_npz(out / "reference.npz", profile=sums[0] / len(tokens),
               common_logits=sums[1] / len(tokens), common_probabilities=sums[2] / len(tokens),
               manifest_digest=np.array(digest(manifest)))


def layer_metrics(logits, actual_weights, reference, common_logits, common_probabilities, common_offsets):
    """All outputs [metric,head,query]; no head is treated as an independent sample.

    Common-offset metrics condition on d=0..D-1. Prefix metrics normalize the
    available past keys, including in the optional bidirectional intervention.
    Neither operation estimates an additive architectural contribution.
    """
    heads, length, _ = logits.shape
    d = common_offsets
    q = np.arange(length)
    # Compute from fp32 captured logits for numerical coherence. Actual attention
    # weights are retained separately for first-key, entropy, and effective keys.
    prefix_probs = softmax(np.where(np.isfinite(logits), logits, -np.inf))
    null = relative_null(reference[:, :length])
    common = logits[..., :d].copy()
    common[:, :d-1] = 0
    common -= common.mean(axis=-1, keepdims=True)
    reference_common = reference[:, None, :d]
    reference_common = reference_common - reference_common.mean(axis=-1, keepdims=True)
    conditional = softmax(common)
    null_conditional = softmax(reference_common)
    baseline_mse = np.mean((common - reference_common)**2, axis=-1)
    corrected_mse = np.mean((common - common_logits[:, :length])**2, axis=-1)
    baseline_prob_mse = np.mean((conditional - null_conditional)**2, axis=-1)
    corrected_prob_mse = np.mean((conditional - common_probabilities[:, :length])**2, axis=-1)
    uniform = (np.arange(length)[None, :] <= q[:, None]) / (q[:, None] + 1)
    # Ratios use log-softmax, not clipping tiny probabilities before taking logs.
    finite = np.where(np.isfinite(logits), logits, -np.inf)
    rowmax = np.max(finite, axis=-1, keepdims=True)
    logp = finite - rowmax - np.log(np.exp(finite-rowmax).sum(axis=-1, keepdims=True))
    logz = np.logaddexp.accumulate(reference[:, :length], axis=-1)
    ratio_error = np.zeros((heads, length))
    ratio_error[:, d:] = np.mean(logp[:, d:, :d] - logp[:, d-1:-1, :d], axis=-1) - (logz[:, d-1:-1] - logz[:, d:])
    p = actual_weights.astype(np.float64)
    # BF16 softmax output need not sum to exactly 1 after rounding.
    p /= p.sum(axis=-1, keepdims=True)
    entropy = -np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny)), axis=-1)
    metrics = np.stack([
        baseline_mse - corrected_mse, baseline_prob_mse - corrected_prob_mse,
        np.abs(prefix_probs-null).sum(axis=-1) / 2, js_divergence(prefix_probs, null),
        np.abs(prefix_probs-uniform).sum(axis=-1) / 2,
        p[..., 0], 1 / np.sum(p*p, axis=-1), entropy, np.triu(p, 1).sum(axis=-1),
        ratio_error, baseline_mse, baseline_prob_mse
    ])
    # Invalid common-support positions are placeholders, never used by inference.
    metrics[[0, 1, 10, 11], :, :d-1] = 0
    if not np.isfinite(metrics).all():
        raise RuntimeError("Non-finite metric; inspect model numerical stability")
    return metrics


def evaluate(config, split="test", mode="causal", shard=0, shards=1, limit=None):
    out, manifest = read_run(config)
    with np.load(out / "reference.npz") as z:
        if str(z["manifest_digest"]) != digest(manifest):
            raise ValueError("Reference belongs to a different manifest")
        reference = {k: z[k] for k in ["profile", "common_logits", "common_probabilities"]}
    reference_hash = digest({k: __import__("hashlib").sha256(v.tobytes()).hexdigest() for k, v in reference.items()})
    run_hash = digest([manifest, reference_hash, mode, METRICS])
    tokens = np.load(out / f"tokens-{split}.npz")["tokens"]
    destination = out / "evaluations" / split / mode
    destination.mkdir(parents=True, exist_ok=True)
    indices = list(range(shard, len(tokens), shards))
    if limit is not None:
        indices = indices[:limit]
    todo = []
    for index in indices:
        path = destination / f"{index:05d}.npz"
        if path.exists():
            with np.load(path) as z:
                if str(z["run_hash"]) != run_hash:
                    raise ValueError("Stale evaluation output. Use a new output directory.")
        else:
            todo.append(index)
    if not todo:
        print("Requested evaluations already complete", flush=True)
        return
    model = InstrumentedModel(config, manifest)
    atomic_json(destination / f"environment-shard-{shard}.json", model.metadata)
    for index in todo:
        curves, layer_summaries = [], []
        def collect(layer, module, query, key, weights, scaling, attention_mode):
            if index == todo[0]:
                model.check_weights(query, key, module, weights, scaling, attention_mode)
            logits = model.distance_logits(module, query, key, scaling)
            values = layer_metrics(logits, weights[0].float().cpu().numpy(),
                reference["profile"][layer], reference["common_logits"][layer],
                reference["common_probabilities"][layer], config["common_offsets"])
            curves.append(values.mean(axis=1))
            e0, e1 = config["early"]
            l0, l1 = config["late"]
            layer_summaries.append(np.stack([values[..., e0:e1].mean(axis=-1),
                                             values[..., l0:l1].mean(axis=-1)], axis=-1))
        begin = time.perf_counter()
        model.forward(tokens[index], collect, mode=mode)
        curve = np.stack(curves).mean(axis=0)
        atomic_npz(destination / f"{index:05d}.npz", curve=curve.astype(np.float32),
                   head_bands=np.stack(layer_summaries).astype(np.float32),
                   run_hash=np.array(run_hash), document_id=np.array(manifest["splits"][split][index]["id"]),
                   metrics=np.array(METRICS))
        print(f"{split}/{mode} {index+1}/{len(tokens)}: {time.perf_counter()-begin:.1f}s", flush=True)


def sanity(config):
    """Real-model positive controls; run before expensive collection."""
    out, manifest = read_run(config)
    model = InstrumentedModel(config, manifest)
    tokens = np.load(out / "tokens-validation.npz")["tokens"][0]
    length = min(64, len(tokens))
    def get(t, shift=0):
        captured = []
        def collect(layer, module, query, key, weights, scaling, mode):
            model.check_weights(query, key, module, weights, scaling, mode)
            captured.append(weights[0].float().cpu().numpy())
        model.forward(t, collect, shift=shift)
        return np.stack(captured)
    baseline = get(tokens[:length])
    short = get(tokens[:length//2])
    shifted = get(tokens[:length], shift=37)
    repeated = get(np.repeat(tokens[0], length))
    prefix_error = float(np.max(np.abs(baseline[..., :length//2, :length//2] - short)))
    shift_error = float(np.max(np.abs(baseline - shifted)))
    # With identical tokens and no special prefix, identical hidden states make
    # post-RoPE scores relative-only at each layer, despite changing normalization.
    ratio_errors = []
    for i in range(3, length-1):
        a = repeated[..., i, i-np.arange(4)]
        b = repeated[..., i+1, i+1-np.arange(4)]
        predicted = 1 - repeated[..., i+1, 0]
        ratio_errors.append(np.max(np.abs(b - a * predicted[..., None])))
    tol = 0.03 if config["dtype"] != "float32" else 2e-4
    report = {"prefix_attention_max_error": prefix_error, "position_shift_attention_max_error": shift_error,
        "repeated_token_identity_max_error": float(max(ratio_errors)), "tolerance": tol,
        "passed": max(prefix_error, shift_error, max(ratio_errors)) <= tol,
        "note": "BF16 tolerance is an implementation check; repeat in float32 for fine effects."}
    atomic_json(out / "sanity.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Sanity control failed; do not launch primary collection")
