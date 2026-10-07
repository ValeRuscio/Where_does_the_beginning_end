"""Full-length validation-input check before fitting; no test data used."""
import argparse
import json
from pathlib import Path
import numpy as np
from .cli import load_config
from .model import InstrumentedModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    source = Path(cfg["output"])
    manifest = json.loads((source / "manifest.json").read_text())
    model = InstrumentedModel(cfg, manifest)
    tokens = np.load(source / "tokens-validation.npz")["tokens"][0]
    rows = []
    def collect(layer, module, query, key, weights, scaling, mode):
        error = model.check_weights(query, key, module, weights, scaling, mode)
        logits = model.distance_logits(module, query, key, scaling)
        support = np.tri(len(tokens), dtype=bool)
        if not np.isfinite(logits[:, support]).all():
            raise RuntimeError(f"Non-finite accessible score at layer {layer}")
        rows.append({"layer": layer, "native_softmax_max_error": error})
    model.forward(tokens, collect)
    report = {"length": len(tokens), "layers": rows, "passed": True,
              "environment": model.metadata}
    Path("logs/full-length-capture.json").write_text(json.dumps(report, indent=2))
    print(f"Full-length capture passed: {len(tokens)} tokens, {len(rows)} layers.")


if __name__ == "__main__":
    main()
