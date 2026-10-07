import csv
import json
from pathlib import Path
import numpy as np
from .core import atomic_json, atomic_npz, bootstrap_curves, holm, paired_summary
from .experiment import METRICS, read_run


def power_table(alpha=0.05, tests=2, desired_power=0.8):
    from scipy.stats import norm
    rows = []
    for n in [128, 256, 512, 768, 1024]:
        detectable = (norm.ppf(1-alpha/(2*tests)) + norm.ppf(desired_power)) / np.sqrt(n)
        rows.append({"n_test_documents": n, "minimum_paired_standardized_effect": float(detectable),
                     "power": desired_power, "family_alpha": alpha, "tests": tests,
                     "method": "Normal-approximation planning, Bonferroni correction; not a guarantee"})
    return rows


def load_curves(config, split, mode):
    out, manifest = read_run(config)
    n = len(manifest["splits"][split])
    curves, hashes = [], set()
    for index, record in enumerate(manifest["splits"][split]):
        path = out / "evaluations" / split / mode / f"{index:05d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Incomplete split: missing {path}. Never analyze an opportunistic partial test set.")
        with np.load(path) as z:
            if str(z["document_id"]) != record["id"] or list(z["metrics"]) != METRICS:
                raise ValueError("Mismatched document or metric definitions")
            curves.append(z["curve"])
            hashes.add(str(z["run_hash"]))
    if len(hashes) != 1:
        raise ValueError("Mixed models/references in evaluation outputs")
    return out, np.stack(curves).astype(np.float64)


def analyze(config, split="test", mode="causal"):
    out, curves = load_curves(config, split, mode)
    destination = out / "analysis" / split / mode
    destination.mkdir(parents=True, exist_ok=True)
    e0, e1 = config["early"]
    l0, l1 = config["late"]
    endpoint_results = []
    contrasts = curves[:, :2, e0:e1].mean(axis=-1) - curves[:, :2, l0:l1].mean(axis=-1)
    for endpoint in range(2):
        result = paired_summary(contrasts[:, endpoint], config["alpha"])
        # Familywise intervals via conservative Bonferroni correction.
        family_result = paired_summary(contrasts[:, endpoint], config["alpha"] / 2)
        result.update({"endpoint": METRICS[endpoint], "familywise_ci": family_result["ci"],
                       "contrast": "mean early correction gain minus mean late correction gain",
                       "early_gain_descriptive": paired_summary(curves[:,endpoint,e0:e1].mean(axis=-1), config["alpha"]),
                       "late_gain_descriptive": paired_summary(curves[:,endpoint,l0:l1].mean(axis=-1), config["alpha"])})
        endpoint_results.append(result)
    adjusted = holm([r["p_two_sided"] for r in endpoint_results])
    for result, p in zip(endpoint_results, adjusted):
        result["p_holm_two_endpoints"] = float(p)
    confirmatory = split == "test" and mode == "causal" and config["n_test"] >= 512
    summary = {"status": "prespecified primary analysis, if protocol was frozen before opening test outputs"
               if confirmatory else "pilot, validation, or exploratory intervention analysis",
        "independent_unit": "source document", "n_documents": len(curves),
        "uncertainty": "Conditional on fixed training reference; documents resampled, heads and positions kept together",
        "early_queries_inclusive": [e0, e1-1], "late_queries_inclusive": [l0, l1-1],
        "primary_endpoints": endpoint_results,
        "power_plan": power_table(config["alpha"]),
        "warning": "A positive gain detects predictable changes in mean common-offset structure. It is not total positional information or a causal source decomposition."}
    atomic_json(destination / "summary.json", summary)
    with np.load(out / "reference.npz") as reference_file:
        f = reference_file["profile"]
    final_distance_profile = np.exp(f - np.max(f, axis=-1, keepdims=True))
    final_distance_profile /= final_distance_profile.sum(axis=-1, keepdims=True)
    tail_mass = 1 - final_distance_profile.cumsum(axis=-1)
    atomic_npz(destination / "reference-tail.npz", remaining_mass=tail_mass,
               q95=(tail_mass <= 0.05).argmax(axis=-1), q99=(tail_mass <= 0.01).argmax(axis=-1))
    estimate, lo, hi, simultaneous_lo, simultaneous_hi = bootstrap_curves(
        curves, config["bootstrap"], config["seed"] + 271, config["alpha"])
    atomic_npz(destination / "curves.npz", metrics=np.array(METRICS), mean=estimate,
               pointwise_low=lo, pointwise_high=hi, simultaneous_low=simultaneous_lo,
               simultaneous_high=simultaneous_hi)
    with (destination / "curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "query", "mean", "pointwise_low", "pointwise_high", "simultaneous_low", "simultaneous_high", "valid"])
        for m, name in enumerate(METRICS):
            for q in range(config["length"]):
                minimum = config["common_offsets"] if m == 9 else config["common_offsets"] - 1 if m in [0,1,10,11] else 0
                writer.writerow([name, q, estimate[m,q], lo[m,q], hi[m,q],
                                 simultaneous_lo[m,q], simultaneous_hi[m,q], q >= minimum])
    length_rows = []
    for length in config["lengths"]:
        for m in [2, 3, 5, 6, 7]:
            x = curves[:, m, :length].mean(axis=-1)
            length_rows.append({"length": length, "metric": METRICS[m], **paired_summary(x, config["alpha"])})
    atomic_json(destination / "nested_length_averages.json", {
        "interpretation": "Descriptive averages of nested prefixes from the SAME causal forward. This cannot show a change at a fixed query when a suffix is appended.",
        "rows": length_rows})
    plot_curves(config, destination, estimate, lo, hi)
    print(json.dumps(summary, indent=2))


def plot_curves(config, destination, mean, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [0, 1, 2, 5, 6, 9]
    labels = ["Held-out logit correction gain", "Held-out conditional-probability correction gain",
              "TV from relative-only prefix prediction", "Attention mass on first key",
              "Effective number of keys", "Matched-offset log-ratio prediction error"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for ax, m, label in zip(axes.flat, selected, labels):
        start = config["common_offsets"] if m == 9 else config["common_offsets"]-1 if m in [0,1] else 0
        x = np.arange(start, config["length"])
        ax.plot(x, mean[m,start:], linewidth=1)
        ax.fill_between(x, lo[m,start:], hi[m,start:], alpha=0.2)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xscale("symlog", linthresh=16)
        ax.set_xlabel("Query position (zero-based)")
        ax.set_title(label, fontsize=10)
    fig.suptitle("Document means and pointwise 95% bootstrap intervals\nSimultaneous bands are available in curves.csv")
    fig.savefig(destination / "boundary-curves.pdf")
    fig.savefig(destination / "boundary-curves.png", dpi=160)
    plt.close(fig)


def compare_masks(config, split="test"):
    out, causal = load_curves(config, split, "causal")
    _, bidirectional = load_curves(config, split, "bidirectional")
    difference = causal - bidirectional
    result = {"interpretation": "Exploratory paired whole-network mask intervention. All hidden states can change. Differences are not an additive causal share.",
              "results": []}
    for metric in [0, 1, 2, 5]:
        a, b = config["early"]
        summary = paired_summary(difference[:,metric,a:b].mean(axis=-1), config["alpha"])
        result["results"].append({"metric": METRICS[metric], **summary})
    p = holm([r["p_two_sided"] for r in result["results"]])
    for r, value in zip(result["results"], p):
        r["p_holm_four_exploratory_tests"] = float(value)
    atomic_json(out / "analysis" / split / "mask-comparison.json", result)
    print(json.dumps(result, indent=2))
