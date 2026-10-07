# Manuscript-to-code map

| Result | Code / saved data | Sample and support |
|---|---|---|
| Figure 1: local gains and first-key mass | `paper_assets/build_figures.py`; `saved_results/predictive/*/test-documents.npz` | 512 test documents; local offsets 0–3; query 4 onward; first mass through query 1023 |
| Table 1: four predictive endpoints | `predictive/*/boundary_attention/revised.py`; `analyze_saved.py` | Local gain queries 4–7 minus 32–63; KL improvement queries 4–255 |
| Relative-only / first-key / current-key predictions | Same revised modules; `paper_assets/data/*/predictions-curves.npz` | Relative profile fitted at queries 512–1023, offsets 0–255; bias fitted at queries 32–255 on fitting documents |
| Table 2 and Figure 2: interventions | `interventions/design.py`, `engine.py`, `experiment.py`, `report.py`; `saved_results/interventions/*/documents.npz` | 128 reused test documents; fixed passage 512 tokens; 4 donors/document |
| Boundary endpoint | `primary_head_endpoints` in `interventions/design.py` | Local-TV response to 64-token prefix: passage queries 4–7 minus 32–63 |
| Replacement endpoint | Same function | Token-0 minus token-1 local-TV response, passage queries 5–63; negative in both models |
| Precision sensitivity | `predictive/qwen25/boundary_attention/precision_math.py`, `precision_endpoints.py`; `paper_assets/data/qwen25/precision-*.json` | 32 validation documents; same fitted float32 references evaluated with native and diagnostic float64 arithmetic |
| Appendix F fixed-BOS trajectories and BOS relocation | `legacy/reported-observations.json` | Earlier descriptive summaries only; raw rerun not supplied |

## Array schemas

Predictive `*-documents.npz` arrays preserve document ordering:

- `document_id`: source article ID, one per document.
- `local`: `[document, metric, query]`; metric order is `LOCAL` in `revised.py`.
- `predictions`: `[document, metric, query]`; order is `PRED` in `revised.py`.
- `head_endpoints`: `[document, layer, endpoint, head]`; first four are the primary endpoints, fifth is late first-key mass.

Intervention `documents.npz` arrays:

- `document_id`, `donor_tokens`, `query_coordinates`, `conditions`: identity/design metadata.
- `boundary_*`: `[document, condition, passage_query]`; conditions are prefix 0, 16, 64, 256, then shuffled 64. Heads/layers are averaged within documents.
- `replacement_*` response arrays: `[document, replaced_position, passage_query]`.
- `replacement_donor_local_tv`: `[document, donor, replaced_position, passage_query]`.
- `head_endpoints`: `[document, endpoint, layer, head]`; `endpoints`: `[document, endpoint]`.
- `position_shift_*`: numerical-control curves or document scalar.

The common passage-query storage axis begins at 4. Replacement inference and plots exclude its first entry, because query 4 would directly include altered key 1. The compact format differs from the per-document GPU output format; `analyze_saved.py` handles this explicitly.

## Earlier descriptive observations

The earlier selected-head BOS sink magnitudes must not be compared numerically with the new all-head/all-layer attention averages. Reported zero standard deviations in the fixed-BOS study have the precision of the original summary; they are not a fresh bitwise-equivalence test. Their inclusion does not create an independently replicated result in a third model family.
