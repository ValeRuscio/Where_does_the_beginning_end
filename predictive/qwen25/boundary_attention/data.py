"""One independently sampled source document per experimental unit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import atomic_json, atomic_npz, digest


def download_documents(destination, count=1400, seed=20260929):
    if count < 1:
        raise ValueError("count must be positive")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    path = Path(destination)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    revision = HfApi().dataset_info("wikimedia/wikipedia").sha
    stream = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                          streaming=True, revision=revision).shuffle(seed=seed, buffer_size=10000)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    tmp = path.with_suffix(".partial")
    with tmp.open("w") as f:
        for row in stream:
            text = row["text"].strip()
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            if len(text) < 16000 or content_hash in seen:
                continue
            seen.add(content_hash)
            f.write(json.dumps({"id": str(row["id"]), "text": text,
                                "source": row.get("url"), "sha256": content_hash}) + "\n")
            if len(seen) % 100 == 0:
                print(f"Collected {len(seen)}/{count} distinct long documents", flush=True)
            if len(seen) == count:
                break
    if len(seen) != count:
        raise RuntimeError("Not enough eligible documents; partial file retained")
    tmp.replace(path)
    atomic_json(path.with_suffix(".source.json"), {
        "dataset": "wikimedia/wikipedia", "configuration": "20231101.en",
        "revision": revision, "seed": seed, "count": count,
        "selection": "streaming buffer shuffle; >=16000 characters; exact-text deduplication",
        "limitation": "English Wikipedia sample; near-duplicate articles are not removed"})


def prepare(config):
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    out = Path(config["output"])
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["config_digest"] != digest(config):
            raise ValueError("Output already belongs to a different config; use a new output directory")
        print("Existing prepared manifest retained. Use a new output directory to change data.")
        return
    revision = HfApi().model_info(config["model"], revision=config["revision"]).sha
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision)
    rows, seen_ids, seen_text = [], set(), set()
    with Path(config["dataset"]).open() as f:
        for line in f:
            row = json.loads(line)
            ident = str(row["id"])
            checksum = hashlib.sha256(row["text"].encode()).hexdigest()
            if ident in seen_ids or checksum in seen_text:
                raise ValueError("Duplicate document id or exact text in input; deduplicate before splitting")
            seen_ids.add(ident)
            seen_text.add(checksum)
            rows.append((digest([config["seed"], ident]), ident, row["text"], checksum))
    rows.sort()
    sizes = [config["n_train"], config["n_validation"], config["n_test"]]
    if len(rows) < sum(sizes):
        raise ValueError(f"Need {sum(sizes)} distinct documents; got {len(rows)}")
    # Choose source IDs BEFORE tokenization. Model comparisons use identical source IDs.
    rows = rows[:sum(sizes)]
    bos_id = tokenizer.bos_token_id if config["bos"] == "native" else None
    if config["bos"] == "native" and bos_id is None:
        raise ValueError("Tokenizer has no native BOS; choose none or design a separate explicit-token condition")
    needed = config["length"] - int(bos_id is not None)
    tokens, records = [], []
    for _, ident, text, checksum in rows:
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Exclude the opening 128 tokens and ending 128 tokens; use a random interior passage.
        if len(ids) < needed + 256:
            raise ValueError(f"Document {ident} is too short for an interior crop. Supply longer input documents.")
        rng = np.random.default_rng(int(digest([config["seed"], ident, "crop"])[:16], 16))
        start = int(rng.integers(128, len(ids) - needed - 128 + 1))
        crop = ids[start:start + needed]
        tokens.append(([bos_id] if bos_id is not None else []) + crop)
        records.append({"id": ident, "text_sha256": checksum, "crop_start_token": start,
                        "source_token_count": len(ids)})
    out.mkdir(parents=True, exist_ok=True)
    begin = 0
    split_records = {}
    for split, size in zip(["train", "validation", "test"], sizes):
        atomic_npz(out / f"tokens-{split}.npz", tokens=np.asarray(tokens[begin:begin+size], dtype=np.int64))
        split_records[split] = records[begin:begin+size]
        begin += size
    atomic_json(out / "manifest.json", {
        "config": config, "config_digest": digest(config), "model_revision": revision,
        "tokenizer_class": type(tokenizer).__name__, "bos_id": bos_id,
        "source_file_sha256": hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest(),
        "splits": split_records,
        "crop_note": "Same source documents across model runs; tokenized passage boundaries can differ."})
    print(f"Prepared {sum(sizes)} documents; pinned model revision {revision}", flush=True)
