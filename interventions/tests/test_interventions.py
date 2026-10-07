import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2Config,Qwen2Model,Qwen3Config,Qwen3Model
from design import boundary_conditions,donor_tokens,primary_head_endpoints,band,tv,validate
from engine import Engine
from experiment import document
from report import holm,summarize,analyze
from run_experiments import atomic_npz,initialize

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'config.json').read_text())

def tiny(kind,zero=False):
    torch.manual_seed(3)
    config_cls,model_cls=(Qwen2Config,Qwen2Model) if kind=='qwen2' else (Qwen3Config,Qwen3Model)
    cfg=config_cls(vocab_size=128,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
                   num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=2048,
                   attention_dropout=0.,rope_theta=1000000.,use_sliding_window=False)
    cfg._attn_implementation='eager'
    model=model_cls(cfg).float().eval()
    if zero:
        with torch.no_grad():
            for p in model.parameters():p.zero_()
    return Engine(model)

class DesignTests(unittest.TestCase):
    def test_exact_suffix_and_shuffle(self):
        x=np.arange(1024)
        conditions=boundary_conditions(x,CFG,'document')
        self.assertEqual([m for _,_,m in conditions],[0,16,64,256,64])
        for _,seq,m in conditions:np.testing.assert_array_equal(seq[m:],x[256:768])
        shuffled=conditions[-1][1]
        self.assertEqual(shuffled[0],x[192]);self.assertEqual(shuffled[63],x[255])
        np.testing.assert_array_equal(np.sort(shuffled[:64]),x[192:256])
        self.assertFalse(np.array_equal(shuffled[:64],x[192:256]))
        np.testing.assert_array_equal(shuffled,boundary_conditions(x,CFG,'document')[-1][1])
    def test_donors_matched_and_exclude_special_and_original(self):
        train=np.tile(np.arange(100)[:,None],(1,1024))
        donors=donor_tokens(train,[3,4],[0,1,2],4,7,'a')
        self.assertEqual(len(set(donors)),4)
        self.assertFalse(set(donors)&{0,1,2,3,4})
        np.testing.assert_array_equal(donors,donor_tokens(train,[3,4],[0,1,2],4,7,'a'))
    def test_measured_local_keys_exclude_interventions(self):
        for q in range(5,512):self.assertFalse(set(q-np.arange(4))&{0,1})
        self.assertIn(1,4-np.arange(4))
    def test_primary_contrast_indices(self):
        b=np.zeros((5,2,3,508));r=np.zeros((2,2,3,508))
        b[2,:,:,:4]=.4;b[2,:,:,28:60]=.1
        r[0,:,:,1:60]=.2;r[1,:,:,1:60]=.05
        r[1,:,:,0]=999 # excluded query 4 must not alter result
        result=primary_head_endpoints(b,r)
        np.testing.assert_allclose(result[0],.3);np.testing.assert_allclose(result[1],.15)
    def test_stats_and_adjustment(self):
        np.testing.assert_allclose(holm([.02,.001,.5,.03]),[.06,.004,.5,.06])
        result=summarize(np.arange(1.,11.),4)
        self.assertEqual(result['mean'],5.5)
        self.assertLess(result['familywise_ci'][0],5.5)
        self.assertGreater(result['familywise_ci'][1],5.5)
        self.assertEqual(summarize([0.,0.],4)['p_two_sided'],1.)
    def test_fixed_protocol(self):
        validate(CFG)
        bad={**CFG,'prefix_lengths':[0,64]}
        with self.assertRaises(ValueError):validate(bad)
    def test_resume_identity_and_changed_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            directory,tokens,ids,identity=initialize(CFG,'pilot',out,'qwen25')
            again=initialize(CFG,'pilot',out,'qwen25')
            self.assertEqual(identity,again[3]);self.assertEqual(tokens.shape,(2,1024))
            with self.assertRaises(RuntimeError):
                initialize({**CFG,'replacement_donors':2},'pilot',out,'qwen25')
    def test_atomic_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz';atomic_npz(p,{'endpoints':np.array([1.,2.])})
            with np.load(p) as z:np.testing.assert_array_equal(z['endpoints'],[1.,2.])
            self.assertFalse(p.with_suffix('.tmp').exists())

class NativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_preflight_both_native_adapters(self):
        tokens=np.random.default_rng(0).integers(4,128,64)
        for kind in ['qwen2','qwen3']:
            with self.subTest(kind=kind):self.assertTrue(tiny(kind).preflight(tokens)['passed'])
    def test_prefix_invariance_and_local_probability(self):
        eng=tiny('qwen3');tokens=np.random.default_rng(1).integers(4,128,96)
        a=eng.collect(tokens[:64],suffix_length=64,retain_full=True,audit=True)
        b=eng.collect(tokens,suffix_length=96)
        np.testing.assert_allclose(a['u'],b['u'][...,:60,:],atol=2e-6)
        weights=np.stack(a['full']);q=np.arange(4,64)
        local=weights[:,:,q[:,None],q[:,None]-np.arange(4)[None,:]]
        conditional=local/local.sum(-1,keepdims=True)
        np.testing.assert_allclose(a['u'],conditional,atol=2e-6)
    def test_hooks_restored_on_exception(self):
        eng=tiny('qwen2');original=eng.module.eager_attention_forward
        def fail(*args):raise ValueError('intentional')
        with self.assertRaises(ValueError):
            with eng.observer(fail):eng.model(torch.tensor([[4,5,6]]),use_cache=False)
        self.assertIs(eng.module.eager_attention_forward,original)
    def test_uniform_attention_has_no_local_boundary_effect(self):
        eng=tiny('qwen2',zero=True)
        rng=np.random.default_rng(9);tokens=rng.integers(4,128,1024);train=rng.integers(4,128,(128,1024))
        cfg={**CFG,'replacement_donors':1}
        result=document(eng,tokens,cfg,'uniform',train,[0,1,2,3],audit=True)
        np.testing.assert_allclose(result['endpoints'],0,atol=1e-9)
        for j,m in enumerate([0,16,64,256,64]):
            expected=1/(np.arange(4,512)+m+1)
            np.testing.assert_allclose(result['boundary_first_mass'][j],np.tile(expected,(2,1)),atol=1e-7)
        self.assertEqual(result['replacement_full_tv'].shape,(2,2,508))
    def test_complete_document_and_analysis(self):
        eng=tiny('qwen3');rng=np.random.default_rng(8)
        tokens=rng.integers(4,128,1024);train=rng.integers(4,128,(128,1024))
        cfg={**CFG,'replacement_donors':2,'models':['qwen3'],'bootstrap_replicates':100}
        result=document(eng,tokens,cfg,'a',train,[0,1,2,3],audit=True)
        self.assertEqual(result['head_endpoints'].shape,(2,2,2))
        self.assertEqual(result['replacement_donor_local_tv'].shape,(2,2,2,508))
        self.assertTrue(np.all(result['replacement_full_tv']>=0))
        with tempfile.TemporaryDirectory() as d:
            stage=Path(d)/'main';directory=stage/'qwen3';(directory/'documents').mkdir(parents=True)
            (directory/'manifest.json').write_text(json.dumps({'document_ids':['a','b','c'],'identity':'example'}))
            for i,doc in enumerate(['a','b','c']):
                r={**result,'document_id':np.asarray(doc),'run_identity':np.asarray('example'),
                   'endpoints':result['endpoints']+i*.0001}
                atomic_npz(directory/'documents'/f'{i:04d}.npz',r)
            with contextlib.redirect_stdout(io.StringIO()):
                summary=analyze(ROOT,cfg,'main',stage)
            self.assertTrue(summary['complete'])
            for filename in ['summary.json','curves.pdf','curves.png','document-endpoints.csv','curve-summary.npz']:
                self.assertTrue((directory/filename).exists())

if __name__=='__main__':unittest.main()
