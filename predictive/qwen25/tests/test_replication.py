import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from boundary_attention.core import atomic_json, atomic_npz, digest
from boundary_attention.replication import prepare, verify_tokenizers


class FakeTokenizer:
    is_fast=True
    all_special_ids=[99]
    def __init__(self, changed=False):
        self.backend={"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]},
            "normalizer":None,"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},
            "added_tokens":[{"id":99,"content":"special","special":True}]}
        if changed:self.backend["model"]["vocab"]["b"]=2
        self.backend_tokenizer=SimpleNamespace(to_str=lambda:json.dumps(self.backend))
    def convert_ids_to_tokens(self,ids):return [str(i) for i in ids]
    def batch_decode(self,sequences,**kwargs):return ["ab" for _ in sequences]


class ReplicationTests(unittest.TestCase):
    def test_tokenizer_equivalence_and_rejections(self):
        a,b=FakeTokenizer(),FakeTokenizer()
        self.assertEqual(verify_tokenizers(a,b,[0,1])["unique_used_ids"],2)
        with self.assertRaisesRegex(ValueError,"components differ"):
            verify_tokenizers(a,FakeTokenizer(changed=True),[0,1])
        with self.assertRaisesRegex(ValueError,"special-token"):
            verify_tokenizers(a,b,[0,99])

    def test_added_rules_must_be_absent_from_every_input(self):
        a,b=FakeTokenizer(),FakeTokenizer()
        b.backend["added_tokens"].append({"id":8,"content":"<think>","special":False})
        result=verify_tokenizers(a,b,[0,1],[[0,1]])
        self.assertEqual(result["differing_added_strings_verified_absent"],["<think>"])
        b.backend["added_tokens"][-1]["content"]="ab"
        with self.assertRaisesRegex(ValueError,"occurs"):
            verify_tokenizers(a,b,[0,1],[[0,1]])

    def test_prepare_preserves_exact_documents_and_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot=root/"input";snapshot.mkdir()
            cfg={"model":"target","revision":"main","dataset":"snapshot","output":str(root/"out"),
                 "seed":1,"length":64,"lengths":[32,64],"n_train":3,"n_validation":3,"n_test":3,
                 "common_offsets":4,"fit_query_min":32,"early":[4,8],"late":[16,32],
                 "bos":"none","dtype":"float32","device":"cpu","bootstrap":24,"alpha":.05}
            src={"config":dict(cfg,model="source"),"model_revision":"source-sha","bos_id":None,
                 "source_file_sha256":"original-corpus-hash","splits":{s:[{"id":f"{s}-{i}"} for i in range(3)] for s in ["train","validation","test"]}}
            atomic_json(snapshot/"manifest.json",src)
            for split in src["splits"]:atomic_npz(snapshot/f"tokens-{split}.npz",tokens=np.zeros((3,64),dtype=np.int64))
            config=root/"config.json";atomic_json(config,cfg)
            with patch("transformers.AutoTokenizer.from_pretrained",return_value=FakeTokenizer()), \
                 patch("huggingface_hub.HfApi.model_info",return_value=SimpleNamespace(sha="target-sha")), \
                 contextlib.redirect_stdout(io.StringIO()):
                prepare(config,snapshot);prepare(config,snapshot)
            out=Path(cfg["output"]);manifest=json.loads((out/"manifest.json").read_text())
            self.assertEqual(manifest["model_revision"],"target-sha")
            self.assertEqual(manifest["splits"],src["splits"])
            self.assertEqual(manifest["config_digest"],digest(cfg))
            for split in src["splits"]:
                self.assertEqual((snapshot/f"tokens-{split}.npz").read_bytes(),(out/f"tokens-{split}.npz").read_bytes())
            self.assertFalse((out/"reference.npz").exists())
            atomic_npz(out/"tokens-test.npz",tokens=np.ones((3,64),dtype=np.int64))
            with self.assertRaisesRegex(ValueError,"match the fixed snapshot"):
                prepare(config,snapshot)


if __name__=="__main__":unittest.main()
