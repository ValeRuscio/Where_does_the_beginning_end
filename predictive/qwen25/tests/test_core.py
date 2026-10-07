import unittest
import numpy as np
from boundary_attention.core import softmax, relative_null, holm, bootstrap_curves, js_divergence
from boundary_attention.experiment import training_features, layer_metrics


class NumericalTests(unittest.TestCase):
    def test_js_with_float32_masked_zeros(self):
        p = np.array([[1, 0, 0], [0.25, 0.75, 0]], dtype=np.float32)
        for dtype in (np.float32, np.float64):
            q = np.array([[0, 1, 0], [0.25, 0.75, 0]], dtype=dtype)
            with np.errstate(divide="raise", invalid="raise"):
                result = js_divergence(p, q)
                same = js_divergence(p, p)
            np.testing.assert_allclose(result, [np.log(2), 0], atol=1e-14)
            np.testing.assert_allclose(same, 0, atol=1e-14)

    def test_relative_normalization_identity(self):
        f = np.array([[0, -0.3, 1.2, -2, 0.4]])
        p = relative_null(f)[0]
        for i in range(4):
            np.testing.assert_allclose(p[i+1,:i+1], p[i,:i+1] * (1-p[i+1,i+1]), atol=1e-14)
        np.testing.assert_allclose(p.sum(axis=-1), 1)

    def test_saturated_profile_recovers_relative_logits_with_row_gauge(self):
        rng = np.random.default_rng(1)
        length = 12
        f = rng.normal(size=(2, length))
        logits = f[:,None,:] + rng.normal(size=(2,length,1))
        logits = np.where(np.arange(length)[None,:] <= np.arange(length)[:,None], logits, np.nan)
        estimated, common, cp = training_features(logits, 6, 4)
        np.testing.assert_allclose(estimated, f-f[:,:1], atol=1e-14)
        p = relative_null(f)
        # Convert [query,distance] to [query,key] for actual observed weights.
        by_key = np.zeros_like(p)
        for q in range(length):
            by_key[:,q,:q+1] = p[:,q,:q+1][:,::-1]
        metrics = layer_metrics(logits, by_key, estimated, common, cp, 4)
        np.testing.assert_allclose(metrics[2], 0, atol=1e-14)
        np.testing.assert_allclose(metrics[9,:,4:], 0, atol=1e-14)
        np.testing.assert_allclose(metrics[:2,:,3:], 0, atol=1e-14)
        np.testing.assert_allclose(metrics[8], 0)

    def test_common_support_cancels_prefix_normalization(self):
        f = np.arange(12) * -0.05
        p = relative_null(f)
        common = p[3:,:4] / p[3:,:4].sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(common, np.broadcast_to(softmax(f[:4]), common.shape))

    def test_position_correction_detects_planted_boundary(self):
        L = 64
        logits = np.zeros((1,L,L))
        logits[:,3:8,0] = 2
        logits = np.where(np.arange(L)[None,:] <= np.arange(L)[:,None], logits, np.nan)
        f, common, cp = training_features(logits, 32, 4)
        p = softmax(np.where(np.isfinite(logits), logits, -np.inf))
        keyp = np.zeros_like(p)
        for q in range(L):
            keyp[:,q,:q+1] = p[:,q,:q+1][:,::-1]
        metrics = layer_metrics(logits, keyp, f, common, cp, 4)
        self.assertGreater(metrics[0,0,3:8].mean(), 0.1)
        self.assertGreater(metrics[1,0,3:8].mean(), 0.01)
        np.testing.assert_allclose(metrics[:2,0,32:64], 0)

    def test_holm(self):
        np.testing.assert_allclose(holm([0.03, 0.001, 0.04]), [0.06, 0.003, 0.06])

    def test_bootstrap_resamples_documents(self):
        rng = np.random.default_rng(8)
        a = rng.normal(size=(100, 1, 5))
        x = np.concatenate([a, a*2], axis=1)
        mean, lo, hi, slo, shi = bootstrap_curves(x, 200, 19)
        np.testing.assert_allclose(mean[1], 2*mean[0])
        np.testing.assert_allclose(lo[1], 2*lo[0])
        self.assertTrue(np.all(slo <= mean))
        self.assertTrue(np.all(shi >= mean))


if __name__ == "__main__":
    unittest.main()
