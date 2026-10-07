"""Offline integration checks: tiny random models, never pretrained findings."""
import importlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from transformers import AutoModel, Qwen3Config, Qwen2Config, LlamaConfig, MistralConfig
from boundary_attention.model import InstrumentedModel


class ModelIntegrationTests(unittest.TestCase):
    def test_capture_check_uses_full_matmul_and_rejects_bad_weights(self):
        observer = InstrumentedModel.__new__(InstrumentedModel)
        observer.torch = torch
        observer.module = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
        module = SimpleNamespace(num_key_value_groups=2)
        generator = torch.Generator().manual_seed(17)
        query = torch.randn(1, 4, 16, 8, generator=generator)
        key = torch.randn(1, 2, 16, 8, generator=generator)
        keys = observer.module.repeat_kv(key, 2)
        logits = torch.matmul(query, keys.transpose(2, 3)) * 8**-0.5
        mask = torch.ones(16, 16, dtype=torch.bool).triu(1)
        weights = torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1)
        with patch.object(torch, "matmul", wraps=torch.matmul) as matmul:
            error = observer.check_weights(query, key, module, weights, 8**-0.5, "causal")
        self.assertLess(error, 2e-5)
        self.assertEqual(matmul.call_args.args[0].shape, query.shape)
        # An interior row that the old three-row check never inspected.
        bad = weights.clone()
        bad[:, :, 8, :9] = 0
        bad[:, :, 8, 0] = 1
        with self.assertRaisesRegex(RuntimeError, "capture mismatch"):
            observer.check_weights(query, key, module, bad, 8**-0.5, "causal")
        bad[:, :, 8, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            observer.check_weights(query, key, module, bad, 8**-0.5, "causal")

    def test_supported_architectures(self):
        for cls in [Qwen3Config, Qwen2Config, LlamaConfig, MistralConfig]:
            with self.subTest(model=cls.__name__):
                self.check_model(cls)

    def check_model(self, cls):
        config = cls(vocab_size=40, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, max_position_embeddings=256, sliding_window=None,
                     attention_dropout=0.0)
        config._attn_implementation = "eager"
        model = AutoModel.from_config(config).eval()
        run = {"seed": 1, "dtype": "float32", "device": "cpu", "model": "tiny-local"}
        with patch("transformers.AutoModel.from_pretrained", return_value=model):
            observer = InstrumentedModel(run, {"model_revision": "local-test"})
        token_ids = np.arange(16) % 40
        ids = torch.tensor(token_ids)[None]
        with torch.inference_mode():
            before = model(input_ids=ids, use_cache=False).last_hidden_state.clone()
            with observer.capture(lambda *args: None):
                after = model(input_ids=ids, use_cache=False).last_hidden_state.clone()
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        def get(tokens, mode="causal", shift=0):
            values, logits = [], []
            def collect(layer, module, query, key, weights, scaling, attention_mode):
                observer.check_weights(query, key, module, weights, scaling, attention_mode)
                values.append(weights[0].float().numpy())
                native = observer._native_capture[3]
                reconstructed = torch.matmul(query, observer.module.repeat_kv(key, module.num_key_value_groups).transpose(2, 3)) * scaling
                valid = torch.ones(native.shape[-2:], dtype=torch.bool).tril()
                torch.testing.assert_close(native[..., valid], reconstructed[..., valid], rtol=0, atol=0)
                with patch.object(torch, "matmul", side_effect=AssertionError("Logits must be observed, not recomputed")):
                    logits.append(observer.distance_logits(module, query, key, scaling))
            observer.forward(tokens, collect, mode, shift)
            return np.stack(values), np.stack(logits)
        original = observer.module.eager_attention_forward
        full, logits = get(token_ids)
        self.assertIs(observer.module.eager_attention_forward, original)
        short, _ = get(token_ids[:8])
        np.testing.assert_allclose(full[...,:8,:8], short, atol=2e-6)
        shifted, _ = get(token_ids, shift=37)
        np.testing.assert_allclose(full, shifted, atol=2e-6)
        bidir, _ = get(token_ids, "bidirectional")
        self.assertGreater(float(np.triu(bidir,1).sum()), 0)
        np.testing.assert_allclose(full.sum(axis=-1), 1, atol=1e-6)
        repeated, _ = get(np.full(16, 3))
        for q in range(3,15):
            a = repeated[..., q, q-np.arange(4)]
            b = repeated[..., q+1, q+1-np.arange(4)]
            np.testing.assert_allclose(b, a*(1-repeated[...,q+1,0])[...,None], atol=2e-6)


if __name__ == "__main__":
    unittest.main()
