import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from transformers import AutoModel, Qwen3Config
from boundary_attention.core import atomic_json, atomic_npz, digest, softmax
from boundary_attention.revised import fit_bias, log_odds, metrics, fit, evaluate, analyze


class RevisedTests(unittest.TestCase):
    def test_bias_recovers_known_parameter(self):
        rng=np.random.default_rng(52)
        f=rng.normal(size=(2,3,16))
        beta=rng.uniform(-3,3,size=(2,3))
        for kind in ["first","self"]:
            odds=log_odds(f,range(4,16),kind)
            mass=np.exp(-np.logaddexp(0,-(odds+beta[...,None]))).mean(-1)
            np.testing.assert_allclose(fit_bias(f,mass,range(4,16),kind),beta,atol=1e-10)

    def test_first_token_does_not_change_local_metric_after_query3(self):
        L,D=32,4
        f=np.linspace(0,-1,16)[None,:]
        raw=np.broadcast_to(np.linspace(0,-2,L),(1,L,L)).copy()
        raw=np.where(np.arange(L)[None,:]<=np.arange(L)[:,None],raw,np.nan)
        local=raw[...,:D].copy();local[:,:D]=0;local-=local.mean(-1,keepdims=True)
        ref={"profile":f[None],"local_logits":local[None],"local_probabilities":softmax(local)[None],
             "stationary_probability":softmax(f[:,:D])[None],"beta_first":np.array([[2.]]),"beta_self":np.array([[0.]])}
        cfg={"local_offsets":4,"max_offset":15}
        actual=np.zeros((1,L,L))
        for q in range(L):actual[:,q,:q+1]=softmax(raw[:,q,:q+1])[:,::-1]
        a,_=metrics(raw,actual,ref,0,cfg)
        changed=raw.copy()
        for q in range(L):changed[:,q,q]+=10
        b,_=metrics(changed,actual,ref,0,cfg)
        np.testing.assert_array_equal(a[:4,:,4:],b[:4,:,4:])

    def test_planted_first_bias_predicts_heldout_prefixes(self):
        L,D=32,4
        f=np.linspace(0,-2,L)[None,:]
        raw=np.broadcast_to(f[:,None,:],(1,L,L)).copy()
        for q in range(L):raw[:,q,q]+=3.0
        raw=np.where(np.arange(L)[None,:]<=np.arange(L)[:,None],raw,np.nan)
        loc=raw[...,:D].copy();loc[:,:D]=0;loc-=loc.mean(-1,keepdims=True)
        ref={"profile":f[None],"local_logits":loc[None],"local_probabilities":softmax(loc)[None],
             "stationary_probability":softmax(f[:,:D])[None],"beta_first":np.array([[3.]]),"beta_self":np.array([[0.]])}
        actual=np.zeros((1,L,L))
        for q in range(L):actual[:,q,:q+1]=softmax(raw[:,q,:q+1])[:,::-1]
        _,p=metrics(raw,actual,ref,0,{"local_offsets":D,"max_offset":31})
        np.testing.assert_allclose(p[1],0,atol=1e-14)
        self.assertGreater(p[3,:,4:].mean(),0)

    def test_test_access_is_explicit(self):
        with self.assertRaisesRegex(ValueError,"locked"):
            evaluate("nonexistent",split="test")
        with self.assertRaisesRegex(ValueError,"locked"):
            analyze("nonexistent",split="test")

    def test_complete_revised_pipeline(self):
        mc=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                      num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=128)
        mc._attn_implementation="eager"
        model=AutoModel.from_config(mc).eval()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"source";out=root/"revised";source.mkdir()
            original={"model":"tiny","seed":7,"device":"cpu","dtype":"float32","length":64,
                      "fit_query_min":32,"common_offsets":4}
            records={s:[{"id":f"{s}-{i}"} for i in range(3)] for s in ["train","validation","test"]}
            manifest={"config":original,"config_digest":digest(original),"model_revision":"local","splits":records}
            atomic_json(source/"manifest.json",manifest)
            rng=np.random.default_rng(71)
            for split in records:atomic_npz(source/f"tokens-{split}.npz",tokens=rng.integers(0,32,size=(3,64)))
            atomic_npz(source/"reference.npz",profile=np.zeros((2,2,64)),common_logits=np.zeros((2,2,64,4)),
                       common_probabilities=np.full((2,2,64,4),.25),manifest_digest=np.array(digest(manifest)))
            cfg={"source_run":str(source),"output":str(out),"max_offset":15,"local_offsets":4,
                 "early":[4,8],"late":[16,24],"bias_queries":[8,16],"comparison_queries":[4,16],
                 "bootstrap":24,"alpha":.05,"seed":4}
            path=root/"revised.json";atomic_json(path,cfg)
            with patch("transformers.AutoModel.from_pretrained",return_value=model),contextlib.redirect_stdout(io.StringIO()):
                fit(path);fit(path)
                evaluate(path);evaluate(path)
                analyze(path)
            summary=json.loads((out/"analysis/validation/summary.json").read_text())
            self.assertEqual(summary["n_documents"],3)
            self.assertEqual(len(summary["endpoints"]),4)
            self.assertTrue((out/"analysis/validation/revised-curves.pdf").exists())
            self.assertFalse((out/"evaluations/test").exists())


if __name__=="__main__":unittest.main()
