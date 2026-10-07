"""Prepare an exact-input Qwen2.5 replication from the saved Qwen3 tokens.

No raw corpus is required. Reuse is conditional on verified ordinary tokenizer
equivalence and identical IDs/representations for every token in the snapshot.
Learned attention references are NEVER transferred between models.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .core import atomic_json, digest
from .cli import load_config


def verify_tokenizers(source, target, used_ids, sequences=None):
    if not source.is_fast or not target.is_fast:
        raise ValueError("Need fast tokenizers to verify the complete ordinary encoding pipeline")
    a=json.loads(source.backend_tokenizer.to_str())
    b=json.loads(target.backend_tokenizer.to_str())
    components=["model","normalizer","pre_tokenizer","decoder"]
    equal={key:a.get(key)==b.get(key) for key in components}
    if not all(equal.values()):
        raise ValueError(f"Cannot reuse token IDs: ordinary tokenizer components differ: {equal}. Do not silently retokenize or replace the sample.")
    specials=set(source.all_special_ids)|set(target.all_special_ids)
    if specials.intersection(used_ids):
        raise ValueError("Saved inputs contain special-token IDs; this exact-input replication requires ordinary tokens only")
    source_tokens=source.convert_ids_to_tokens(used_ids)
    target_tokens=target.convert_ids_to_tokens(used_ids)
    if source_tokens!=target_tokens or any(t is None for t in source_tokens):
        raise ValueError("At least one used token ID maps to a different token")
    # Added-token differences are allowed only for special tokens absent from inputs.
    ordinary_a=[v for v in a.get("added_tokens",[]) if not v.get("special",False)]
    ordinary_b=[v for v in b.get("added_tokens",[]) if not v.get("special",False)]
    absent_rules=[]
    if ordinary_a!=ordinary_b:
        rules_a={digest(v):v for v in ordinary_a};rules_b={digest(v):v for v in ordinary_b}
        differences=[(rules_a.get(k) or rules_b[k]) for k in rules_a.keys()^rules_b.keys()]
        if sequences is None:
            raise ValueError("Added-token rules differ: need every saved sequence to verify that differing strings are absent")
        absent_rules=sorted({v["content"] for v in differences})
        if {v["id"] for v in differences}.intersection(used_ids):
            raise ValueError("A differing added-token rule is used by the snapshot")
        # Qwen3 adds think/tool-response delimiters. They are irrelevant only if
        # absent from ALL saved passages; do not assume this from model names.
        text=source.batch_decode(sequences,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if any(tag in passage for tag in absent_rules for passage in text):
            raise ValueError("A differing added-token string occurs in a saved passage")
    return {"components_equal":equal,"unique_used_ids":len(used_ids),
        "ordinary_pipeline_digest":digest({key:a.get(key) for key in components}),
        "used_id_mapping_digest":digest(list(zip(used_ids,source_tokens))),
        "special_tokens_in_inputs":False,
        "differing_added_strings_verified_absent":absent_rules,
        "note":"Ordinary BPE pipelines and used token IDs match; chat templates and verified-absent added tokens may differ."}


def prepare(config_path, snapshot="input_snapshot"):
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    cfg=load_config(config_path)
    snapshot=Path(snapshot);out=Path(cfg["output"])
    src=json.loads((snapshot/"manifest.json").read_text())
    source_cfg=src["config"]
    for name in ["length","n_train","n_validation","n_test","seed","bos"]:
        if cfg[name]!=source_cfg[name]:raise ValueError(f"Replication changes frozen input setting: {name}")
    if cfg["bos"]!="none" or src["bos_id"] is not None:raise ValueError("Expected no inserted BOS")
    source_ids=[r["id"] for records in src["splits"].values() for r in records]
    if len(source_ids)!=len(set(source_ids)):raise ValueError("Source document split is not disjoint")
    arrays={};checksums={}
    for split in ["train","validation","test"]:
        file=snapshot/f"tokens-{split}.npz"
        arrays[split]=np.load(file)["tokens"]
        if arrays[split].shape!=(len(src["splits"][split]),cfg["length"]):raise ValueError("Invalid saved token shape")
        checksums[split]=hashlib.sha256(file.read_bytes()).hexdigest()
    fingerprint=digest([src,checksums])
    if (out/"manifest.json").exists():
        existing=json.loads((out/"manifest.json").read_text())
        if existing["config_digest"]!=digest(cfg) or existing.get("replication_source_digest")!=fingerprint:
            raise ValueError("Changed config or input snapshot; use a new output directory")
        for split,checksum in checksums.items():
            if hashlib.sha256((out/f"tokens-{split}.npz").read_bytes()).hexdigest()!=checksum:
                raise ValueError("Prepared tokens do not match the fixed snapshot")
        print("Prepared replication retained; exact input hashes verified.",flush=True)
        return
    revision=HfApi().model_info(cfg["model"],revision=cfg["revision"]).sha
    print("Checking tokenizer compatibility; no model weights are loaded in this step.",flush=True)
    source_tok=AutoTokenizer.from_pretrained(source_cfg["model"],revision=src["model_revision"],trust_remote_code=False)
    target_tok=AutoTokenizer.from_pretrained(cfg["model"],revision=revision,trust_remote_code=False)
    ids=np.unique(np.concatenate([v.reshape(-1) for v in arrays.values()])).tolist()
    check=verify_tokenizers(source_tok,target_tok,ids,np.concatenate(list(arrays.values()),axis=0))
    out.mkdir(parents=True,exist_ok=True)
    for split in arrays:
        destination=out/f"tokens-{split}.npz"
        temp=destination.with_suffix(".tmp")
        shutil.copyfile(snapshot/f"tokens-{split}.npz",temp);temp.replace(destination)
    result={"config":cfg,"config_digest":digest(cfg),"model_revision":revision,
        "tokenizer_class":type(target_tok).__name__,"bos_id":None,
        "source_file_sha256":src["source_file_sha256"],"splits":copy.deepcopy(src["splits"]),
        "crop_note":"Exact saved Qwen3 token sequences reused after ordinary-tokenizer and used-ID equivalence checks.",
        "replication_source_digest":fingerprint,"input_file_sha256":checksums,
        "source_model":source_cfg["model"],"source_model_revision":src["model_revision"],
        "tokenizer_verification":check}
    atomic_json(out/"manifest.json",result)
    atomic_json(out/"tokenizer-verification.json",check)
    print(f"Ready: {cfg['model']} at {revision}; {len(source_ids)} identical source sequences, no Qwen3 attention parameters copied.",flush=True)


def compare(revised_config, primary="primary_results"):
    cfg=json.loads(Path(revised_config).read_text());out=Path(cfg["output"])
    a=json.loads((Path(primary)/"summary.json").read_text())
    b=json.loads((out/"analysis/test/summary.json").read_text())
    if a["n_documents"]!=512 or b["n_documents"]!=512:raise ValueError("Expected both complete 512-document tests")
    if [r["endpoint"] for r in a["endpoints"]]!=[r["endpoint"] for r in b["endpoints"]]:
        raise ValueError("Endpoint definitions differ")
    rows=[]
    for x,y in zip(a["endpoints"],b["endpoints"]):
        rows.append({"endpoint":x["endpoint"],"qwen3_mean":x["mean"],"qwen25_mean":y["mean"],
            "qwen25_familywise_ci":y["familywise_ci"],"qwen25_p_holm":y["p_holm_four_endpoints"],
            "same_positive_direction":bool(x["mean"]>0 and y["mean"]>0)})
    report={"interpretation":"Model replication on identical documents, not an independent corpus replication. No pooling of document counts across models.",
        "endpoints":rows,"qwen3_model_fit":a["model_fit"],"qwen25_model_fit":b["model_fit"],
        "limits":"Logit scales, model sizes, training and architectures differ; do not interpret a cross-model difference as an isolated architecture effect."}
    atomic_json(out/"analysis/test/model-comparison.json",report)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","compare"])
    p.add_argument("--config",default="configs/replication-qwen25.json")
    p.add_argument("--snapshot",default="input_snapshot")
    p.add_argument("--primary",default="primary_results")
    a=p.parse_args()
    if a.command=="prepare":prepare(a.config,a.snapshot)
    else:compare(a.config,a.primary)


if __name__=="__main__":main()
