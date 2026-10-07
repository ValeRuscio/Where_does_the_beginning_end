# Third-party assets and attribution

## Text and token snapshots

The token arrays are cropped and tokenized derivatives of English Wikipedia articles from the `20231101.en` configuration of [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia). The [dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/main/README.md) lists **CC BY-SA 3.0** and **GFDL**. Attribution to article contributors is available through the article links/history pages in `data/ARTICLE_ATTRIBUTION.tsv`. Cropping and tokenization are modifications; the arrays contain text-derived content, not newly authored articles. The precise dataset repository revision is not available in the archived metadata.

License references: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html). Original text remains subject to its applicable attribution and share-alike terms; no study-specific license overrides them.

## Models and libraries

Pretrained weights are not included. They are retrieved from [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) and [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), using the fixed commits in the configurations. Their model cards list Apache-2.0. See the upstream repositories for applicable model terms.

The implementation uses PyTorch, Hugging Face Transformers, NumPy, SciPy and Matplotlib. Dependencies are installed separately; their own licenses apply. Native attention functions are invoked from Transformers 4.57.1. The supplied observers and analysis do not redistribute model weights or replace upstream attribution.

No new license for study-specific code has been assigned in this anonymous review distribution. This file records third-party provenance and is not a license grant over those assets.
