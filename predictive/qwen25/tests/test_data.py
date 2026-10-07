import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from boundary_attention.data import prepare


class FakeTokenizer:
    bos_token_id = None
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [len(word) % 30 for word in text.split()]


class DataTests(unittest.TestCase):
    def test_splits_are_disjoint_reproducible_and_manifest_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "documents.jsonl"
            source.write_text("".join(json.dumps({"id": str(i), "text": str(i)+" word"*600})+"\n" for i in range(12)))
            c = {"output": str(root/"run1"), "dataset": str(source), "model": "dummy", "revision": "main",
                 "n_train": 3, "n_validation": 3, "n_test": 3, "bos": "none", "seed": 71, "length": 64}
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), \
                 patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha="fixed-commit")):
                prepare(c)
                prepare(c)
                c2 = dict(c, output=str(root/"run2"), model="other-model")
                prepare(c2)
            m1 = json.loads((root/"run1/manifest.json").read_text())
            m2 = json.loads((root/"run2/manifest.json").read_text())
            ids = [r["id"] for split in m1["splits"].values() for r in split]
            self.assertEqual(len(set(ids)), 9)
            self.assertEqual(m1["splits"], m2["splits"])
            self.assertEqual(m1["model_revision"], "fixed-commit")
            self.assertIsNone(m1["bos_id"])
            for split in m1["splits"]:
                self.assertEqual(np.load(root/f"run1/tokens-{split}.npz")["tokens"].shape, (3,64))
            with self.assertRaises(ValueError):
                prepare(dict(c, seed=99))


if __name__ == "__main__":
    unittest.main()
