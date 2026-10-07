# Supplementary code: A Long Goodbye

This repository accompanies **A Long Goodbye: Local Position Effects and Persistent Attention to the First Token**. It contains the predictive attention study, the boundary-relocation and token-replacement experiments, fixed tokenized inputs, document-level measurements, and figure-generation code.

There are two reproduction routes: inspect and reanalyze the saved results without a GPU, or rerun the pretrained models. No editable package installation is required. Commands below are run from the repository root with Python 3.10 or newer.

## 1. Reproduce the reported statistics and figures without a GPU

```bash
python -m pip install -r requirements-analysis.txt
python reproduce.py verify
python reproduce.py analyze
python reproduce.py figures
```

`verify` checks the distributed files against their SHA-256 checksums. `analyze` independently recomputes all 20 predictive validation/test and intervention endpoint estimates and familywise intervals from document-level measurements, checks them against the saved reports, and verifies document identities and matched donors. It writes `generated/recomputed-endpoints.json`.

`figures` regenerates the manuscript plots and numerical table rows in `paper_assets/figures/` and `paper_assets/*.tex`. Plotting uses the saved aggregate curves and their original bootstrap bands; it does not refit the language models or regenerate the raw measurements. Appendix F's earlier descriptive tables are documented separately in `RESULTS_MAP.md`.

## 2. Set up for model runs

Use an environment with a working CUDA-enabled PyTorch installation. The recorded Qwen2.5 and intervention runs used Python 3.10, PyTorch 2.6.0+cu124, Transformers 4.57.1, and an NVIDIA A100-SXM4-40GB. Dependency files deliberately do not request or replace PyTorch.

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pip install -r requirements-gpu.txt
python reproduce.py test
```

The CPU tests use tiny randomly initialized Qwen models and synthetic controls; they are implementation tests, not paper results. The distribution was checked with Python 3.12, PyTorch 2.14.0 and Transformers 4.57.1 on CPU. Successful CPU checks do not establish identical GPU arithmetic. Collection retains runtime numerical checks.

Pretrained weights and tokenizers are downloaded on first use; the public Qwen checkpoints ordinarily require no login. No weights are distributed here. The main experiments use float32, native eager attention, no KV cache, batch size one, and fixed model weights. Run one collection process at a time. The Qwen2.5 float64 audit requires at least 18 GiB of free GPU memory. Full collection takes substantially longer than the small implementation tests; use pilot timings on the target hardware rather than assuming a fixed completion time.

## 3. Rerun the predictive study

```bash
python reproduce.py predictive --model qwen3
python reproduce.py predictive --model qwen25
```

Each command prepares the exact saved inputs, performs numerical checks, fits references on 256 fitting documents, fits the first-key/current-key biases, evaluates 512 test documents, and analyzes the results. Qwen2.5 also runs the documented float64 sensitivity procedure before test evaluation. To evaluate the 128-document validation split instead, append `--split validation`.

The stages can be invoked individually, in this order:

```bash
python reproduce.py predictive --model qwen25 --stage prepare
python reproduce.py predictive --model qwen25 --stage fit
python reproduce.py predictive --model qwen25 --stage evaluate
python reproduce.py predictive --model qwen25 --stage analyze
```

Run artifacts are written under `predictive/<model>/runs/`. The fitting and evaluation routines checkpoint progress. Do not change configurations inside an existing run directory. These commands use the revised analysis in `boundary_attention.revised`; the older generic `cli evaluate` metrics are not the revised paper endpoints.

**Preserved implementation difference.** The Qwen3 run used its archived query/key-based score capture; the later Qwen2.5 run records the actual native softmax input. Their packages are deliberately separate. The Qwen3 archive supplied `model.py`, `core.py`, and `revised.py`; compatible preparation and reference-fitting support is supplied from the replication package. The new prepared-manifest hashes differ from the historical runs because repository-relative paths and explicit checkpoint revisions replace the original setup paths. This is a fresh rerun of the same protocol, not reuse of historical fitted-reference hashes or a promise of bitwise-identical GPU output. Saved document-level results provide the direct numerical record.

The strict Qwen2.5 position-shift check can fail in float32. Its audited route preserves that failure, diagnoses it in float64, and checks four actual endpoints on 32 validation documents against a fixed 1% RMS budget. Do not bypass a failed audit by increasing the tolerance. Details are in the precision modules and the manuscript appendix.

## 4. Rerun the paired interventions

```bash
python reproduce.py interventions --stage pilot
python reproduce.py interventions --stage main
```

The pilot evaluates two validation documents per model. Main collection evaluates the first 128 previously examined test documents in their saved order; it is explicitly an **exploratory extension**, not a new independent confirmatory sample. Edit `interventions/config.json` before the pilot to change the model list or document count; changed settings no longer reproduce the reported sample.

- **Boundary relocation:** hold tokens 256–767 fixed and prepend 0, 16, 64, or 256 preceding tokens. Include a shuffled 64-token prefix preserving its first and last tokens.
- **Token replacement:** replace token 0 or token 1 of the unchanged 512-token passage with the same four donor identities, under normal causal attention. Replacement analysis begins at query 5 so neither altered token belongs to the four-offset local support.
- **Numerical control:** shift all position IDs by 64 without changing tokens or causal support; retain the resulting arithmetic sensitivity rather than relabeling it as a context intervention.

Full definitions are in `interventions/PROTOCOL.txt`. The main run checkpoints each source document. Outputs appear in `interventions/results/main/`; the final downloadable archive is `interventions/results/main-intervention-results.zip`. Numerical flags requiring review remain in the summaries.

For a notebook interface, open `interventions/Run_Interventions.ipynb` with the working GPU kernel. It includes setup, tests, pilot, background launch and progress checks. To launch or resume the main run detached from a browser connection after the pilot:

```bash
cd interventions
python launch.py
```

The GPU instance must remain running. Closing the browser is different from shutting down the instance.

## Inputs, analysis units, and uncertainty

The fixed snapshot comprises one 1,024-token passage from each of 896 distinct English Wikipedia source articles: 256 fitting, 128 validation, and 512 test documents. No BOS or chat template is inserted. Both models use identical token IDs, verified for the saved inputs. The replication preparation checks ordinary-tokenizer equivalence and all used token IDs.

The independent unit is the source document. Heads, query positions, and replacement donors are averaged within documents, not treated as additional independent samples. Predictive intervals condition on the fitted references and use four-test families separately within each model. Intervention intervals cover a separate four-test family across both models. Curves have pointwise bootstrap bands, not simultaneous guarantees. The two models do not double the number of independent source documents. See `RESULTS_MAP.md` for support and sample sizes.

`data/ARTICLE_ATTRIBUTION.tsv` lists article IDs and source links. The precise corpus repository revision was absent from the archived preparation metadata; reproducibility relies on the included token arrays and hashes, not a fresh download. The snapshot configuration is `wikimedia/wikipedia`, `20231101.en`. A current article URL is attribution, not a guarantee of the historical text version.

## What is and is not included

- Collection code, pinned model revisions, tests, input arrays, source IDs, figure code, aggregate summaries, and document-level measurements are included.
- Intervention response curves are compacted by averaging layers within each document; donor curves and primary head-level endpoints are retained. This preserves the reported document-level statistics. Full quadratic attention tensors are not distributed.
- Appendix F retains earlier Llama/Mistral fixed-BOS and BOS-relocation summaries. Their raw per-prompt arrays, selected layer IDs for the BOS table, and exact checkpoint revisions were unavailable. This repository does **not** claim a complete rerun of those earlier observations.
- No mask-share decomposition, original topic-probe claim, or length power law is established by the new experiments.

See `THIRD_PARTY_NOTICES.md` for dataset and dependency attribution. No new license for the study-specific code is assigned by this review supplement. Model and data licenses remain those of their providers.
