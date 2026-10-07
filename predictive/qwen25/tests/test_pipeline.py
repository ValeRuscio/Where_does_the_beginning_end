import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from transformers import AutoModel, Qwen3Config
from boundary_attention.core import atomic_json, atomic_npz, digest
from boundary_attention.experiment import fit, evaluate, sanity
from boundary_attention.analysis import analyze, compare_masks


class PipelineTests(unittest.TestCase):
    def test_complete_offline_pipeline_and_resume(self):
        model_config = Qwen3Config(vocab_size=40, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=256, attention_dropout=0)
        model_config._attn_implementation = "eager"
        model = AutoModel.from_config(model_config).eval()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            config = {"model": "tiny", "revision": "local", "dataset": "local",
                "output": str(out), "seed": 7, "length": 64, "lengths": [32,64],
                "n_train": 3, "n_validation": 3, "n_test": 4, "common_offsets": 4,
                "fit_query_min": 32, "early": [3,8], "late": [32,64], "bos": "none",
                "dtype": "float32", "device": "cpu", "bootstrap": 32, "alpha": 0.05}
            records = {}
            rng = np.random.default_rng(11)
            for split, n in [("train",3), ("validation",3), ("test",4)]:
                atomic_npz(out/f"tokens-{split}.npz", tokens=rng.integers(0,40,(n,64)))
                records[split] = [{"id": f"{split}-{i}"} for i in range(n)]
            manifest = {"config": config, "config_digest": digest(config),
                        "model_revision": "local", "splits": records}
            atomic_json(out/"manifest.json", manifest)
            with patch("transformers.AutoModel.from_pretrained", return_value=model), contextlib.redirect_stdout(io.StringIO()):
                sanity(config)
                fit(config)
                with np.load(out/"reference.npz") as z:
                    before = z["profile"].copy()
                fit(config)
                # npz can contain timestamp metadata; compare actual fitted arrays.
                with np.load(out/"reference.npz") as z:
                    self.assertTrue(np.isfinite(z["profile"]).all())
                    np.testing.assert_array_equal(before, z["profile"])
                evaluate(config, limit=1)
                with self.assertRaises(FileNotFoundError):
                    analyze(config)
                evaluate(config, shard=0, shards=2)
                evaluate(config, shard=1, shards=2)
                evaluate(config)
                analyze(config)
                evaluate(config, mode="bidirectional")
                compare_masks(config)
            summary = json.loads((out/"analysis/test/causal/summary.json").read_text())
            self.assertEqual(summary["n_documents"], 4)
            self.assertTrue((out/"analysis/test/causal/boundary-curves.pdf").exists())
            self.assertTrue((out/"analysis/test/mask-comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
