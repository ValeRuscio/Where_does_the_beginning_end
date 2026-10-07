import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import AutoModel, Qwen2Config
from boundary_attention.audited_sanity import eligible
from boundary_attention.precision_endpoints import assess, endpoints
from boundary_attention.precision_math import precise_rope, collect
from boundary_attention.model import InstrumentedModel
from boundary_attention.core import atomic_json, atomic_npz, digest, softmax
from boundary_attention.revised import context
from boundary_attention.precision_endpoints import main as precision_main


class PrecisionTests(unittest.TestCase):
    def test_complete_precision_audit_and_resume(self):
        mc = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        attention_dropout=0, sliding_window=None)
        mc._attn_implementation = "eager"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source"; out = root / "revised"
            original = dict(model="tiny", dtype="float32", device="cpu", seed=7,
                            length=64, fit_query_min=32, common_offsets=4)
            manifest = dict(config=original, config_digest=digest(original), model_revision="tiny",
                            splits={"validation": [{"id": str(i)} for i in range(32)]})
            atomic_json(source / "manifest.json", manifest)
            tokens = np.random.default_rng(1).integers(0, 32, size=(32, 64))
            atomic_npz(source / "tokens-validation.npz", tokens=tokens)
            template = np.broadcast_to(np.array([-.2, -.1, .1, .2]), (2, 2, 64, 4)).copy()
            template[:, :, 4:8] *= 2
            probabilities = softmax(template)
            atomic_npz(source / "reference.npz", profile=np.zeros((2, 2, 64)),
                       common_logits=template, common_probabilities=probabilities,
                       manifest_digest=np.array(digest(manifest)))
            cfg = dict(source_run=str(source), output=str(out), max_offset=15, local_offsets=4,
                       early=[4, 8], late=[16, 24], comparison_queries=[4, 16], bias_queries=[8, 16])
            path = root / "config.json"; atomic_json(path, cfg)
            identity = context(path)[-1]
            atomic_npz(out / "reference.npz", identity=np.array(identity), profile=np.zeros((2, 2, 16)),
                       local_logits=template, local_probabilities=probabilities,
                       stationary_probability=np.full((2, 2, 4), .25),
                       beta_first=np.ones((2, 2)), beta_self=np.full((2, 2), .2))
            with patch("sys.argv", ["precision", "--config", str(path)]), \
                 patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **kw: AutoModel.from_config(mc).eval()) as loader, \
                 contextlib.redirect_stdout(io.StringIO()):
                precision_main()
                self.assertEqual(loader.call_count, 2)
                precision_main()
                self.assertEqual(loader.call_count, 2)
            report = json.loads((out / "precision-sensitivity/summary.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["n_documents"], 32)
            self.assertEqual(len(list((out / "precision-sensitivity/float64").glob("*.npz"))), 32)

    def test_exception_cannot_hide_other_failed_or_nonfinite_controls(self):
        report = dict(prefix_attention_max_error=1e-5, position_shift_attention_max_error=.002,
                      repeated_token_identity_max_error=1e-5, tolerance=.0002)
        self.assertTrue(eligible(report))
        for name in ("prefix_attention_max_error", "repeated_token_identity_max_error"):
            self.assertFalse(eligible(dict(report, **{name: .003})))
        self.assertFalse(eligible(dict(report, position_shift_attention_max_error=np.nan)))

    def test_endpoint_budget_stops_large_changes(self):
        values = np.ones((32, 4))
        self.assertTrue(assess(values * 1.001, values)["passed"])
        self.assertFalse(assess(values * 1.02, values)["passed"])
        values[0, 0] = np.nan
        with self.assertRaises(ValueError): assess(values, values)

    def test_float64_capture_and_endpoint_path(self):
        torch.manual_seed(17)
        config = Qwen2Config(vocab_size=40, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                            max_position_embeddings=128, attention_dropout=0, sliding_window=None)
        config._attn_implementation = "eager"
        model = AutoModel.from_config(config).eval()
        with patch("transformers.AutoModel.from_pretrained", return_value=model):
            observer = InstrumentedModel(dict(model="tiny", dtype="float32", device="cpu", seed=17),
                                         dict(model_revision="test"))
        tokens = np.arange(32) % 40
        ref = dict(profile=np.zeros((2, 2, 16)), local_logits=np.full((2, 2, 32, 4), .2),
                   local_probabilities=np.full((2, 2, 32, 4), .15),
                   stationary_probability=np.full((2, 2, 4), .25),
                   beta_first=np.ones((2, 2)), beta_self=np.full((2, 2), .2))
        cfg = dict(local_offsets=4, max_offset=15, early=[4, 8], late=[16, 24], comparison_queries=[4, 16])
        native = endpoints(observer, tokens, ref, cfg)
        model.double()
        with precise_rope(model, full=True):
            precise = endpoints(observer, tokens, ref, cfg)
            a, b = collect(model, tokens), collect(model, tokens, 37)
            def check(layer, module, query, key, weights, scaling, mode):
                self.assertEqual(observer.distance_logits(module, query, key, scaling).dtype, np.float64)
                self.assertEqual(weights.dtype, torch.float64)
            observer.forward(tokens, check)
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        np.testing.assert_allclose(native, precise, atol=2e-7, rtol=1e-5)


if __name__ == "__main__": unittest.main()
