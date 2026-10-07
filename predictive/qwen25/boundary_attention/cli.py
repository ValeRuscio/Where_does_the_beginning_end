import argparse
import json
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"model", "revision", "dataset", "output", "seed", "length", "lengths", "n_train",
                "n_validation", "n_test", "common_offsets", "fit_query_min", "early", "late",
                "bos", "dtype", "device", "bootstrap", "alpha"}
    if set(config) != required:
        raise ValueError(f"Config keys mismatch: missing {required-set(config)}, extra {set(config)-required}")
    L, D = config["length"], config["common_offsets"]
    if not 2 <= D <= L or not D <= config["fit_query_min"] < L:
        raise ValueError("Invalid length, common support, or reference-fitting query range")
    if not D-1 <= config["early"][0] < config["early"][1] <= config["late"][0] < config["late"][1] <= L:
        raise ValueError("Nonoverlapping early/late bands must have complete common support")
    if max(config["lengths"]) > L or min(config["lengths"]) < 1:
        raise ValueError("Length sweep must stay within the collected prefix")
    if min(config["n_train"], config["n_validation"], config["n_test"]) < 3:
        raise ValueError("Each split needs >=3 documents")
    if config["bos"] not in {"none", "native"}:
        raise ValueError("bos must be none or native")
    return config


def main():
    p = argparse.ArgumentParser(description="Boundary attention experiments; see README and PROTOCOL.md")
    sub = p.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download-data")
    download.add_argument("--output", default="data/documents.jsonl")
    download.add_argument("--count", type=int, default=1400)
    download.add_argument("--seed", type=int, default=20260929)
    sub.add_parser("power")
    for name in ["prepare", "sanity", "fit", "evaluate", "analyze", "compare-masks"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        if name in ["evaluate", "analyze", "compare-masks"]:
            cmd.add_argument("--split", choices=["validation", "test"], default="test")
        if name in ["evaluate", "analyze"]:
            cmd.add_argument("--mode", choices=["causal", "bidirectional"], default="causal")
        if name == "evaluate":
            cmd.add_argument("--shard", type=int, default=0)
            cmd.add_argument("--shards", type=int, default=1)
            cmd.add_argument("--limit", type=int, help="Timing pilot only; analyze refuses incomplete splits")
    args = p.parse_args()
    if args.command == "download-data":
        from .data import download_documents
        download_documents(args.output, args.count, args.seed)
        return
    if args.command == "power":
        from .analysis import power_table
        print(json.dumps(power_table(), indent=2))
        return
    config = load_config(args.config)
    if args.command == "prepare":
        from .data import prepare
        prepare(config)
    elif args.command in ["sanity", "fit", "evaluate"]:
        from .experiment import sanity, fit, evaluate
        if args.command == "sanity":
            sanity(config)
        elif args.command == "fit":
            fit(config)
        else:
            if args.shards < 1 or not 0 <= args.shard < args.shards:
                p.error("Require 0 <= shard < shards")
            evaluate(config, args.split, args.mode, args.shard, args.shards, args.limit)
    else:
        from .analysis import analyze, compare_masks
        if args.command == "analyze":
            analyze(config, args.split, args.mode)
        else:
            compare_masks(config, args.split)


if __name__ == "__main__":
    main()
