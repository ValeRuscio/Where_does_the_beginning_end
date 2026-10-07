"""Numerical definitions, independent of torch and model implementations."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def relative_null(profile):
    """profile [...,d] -> probabilities [...,query,d]. d is distance, not key."""
    length = profile.shape[-1]
    logits = np.broadcast_to(profile[..., None, :], profile.shape[:-1] + (length, length))
    support = np.arange(length)[None, :] <= np.arange(length)[:, None]
    return softmax(np.where(support, logits, -np.inf))


def js_divergence(p, q):
    # NumPy 1.x can cast the float64 floor to zero for float32 inputs.
    # Promote before taking logs so masked zero probabilities contribute zero.
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = (p + q) / 2
    tiny = np.finfo(np.float64).tiny
    return 0.5 * np.sum(p * (np.log(np.maximum(p, tiny)) - np.log(np.maximum(m, tiny)))
                        + q * (np.log(np.maximum(q, tiny)) - np.log(np.maximum(m, tiny))), axis=-1)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ordered = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty_like(p)
    result[order] = np.minimum(ordered, 1)
    return result


def paired_summary(values, alpha=0.05):
    """Document-level t interval/test. scipy is loaded only for inference."""
    from scipy.stats import t
    x = np.asarray(values, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Need at least three finite independent document contrasts")
    n = len(x)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        p = 1.0 if mean == 0 else 0.0
        stat = None
    else:
        stat = mean / se
        p = float(2 * t.sf(abs(stat), n - 1))
    margin = float(t.ppf(1 - alpha / 2, n - 1) * se)
    return {"n_documents": n, "mean": mean, "sd": sd,
            "ci": [mean - margin, mean + margin], "p_two_sided": p,
            "t": stat, "paired_standardized_effect": mean / sd if sd else None}


def bootstrap_curves(curves, repetitions, seed, alpha=0.05):
    """Resample entire documents jointly across metrics and positions.

    A simultaneous band uses the maximum standardized deviation over all
    supplied metrics/positions. It is conditional on the fitted training model.
    """
    x = np.asarray(curves, dtype=np.float64)
    n = x.shape[0]
    if n < 3 or not np.isfinite(x).all():
        raise ValueError("Bootstrap needs >=3 documents and finite curves")
    rng = np.random.default_rng(seed)
    estimate = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(n)
    draws = np.empty((repetitions,) + estimate.shape, dtype=np.float32)
    flat = x.reshape(n, -1)
    for begin in range(0, repetitions, 64):
        end = min(begin + 64, repetitions)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=end-begin) / n
        draws[begin:end] = (weights @ flat).reshape((end-begin,) + estimate.shape)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    active = se > 1e-12
    maxima = np.max(np.where(active, np.abs(draws - estimate) / np.maximum(se, 1e-12), 0)
                    .reshape(repetitions, -1), axis=1)
    critical = float(np.quantile(maxima, 1 - alpha))
    return estimate, lo, hi, estimate - critical * se, estimate + critical * se
