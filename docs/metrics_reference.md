# Metrics reference

This document explains every CSV produced by the pipeline and how the logged
numbers map to the tables in the dissertation *"Text-Image Semantic IDs for
Generative Recommendation"*.

## Where the CSVs come from

Every GRID command writes its logs under `<GRID_ROOT>/logs/`:

| Stage | Log location |
|---|---|
| semantic-embedding inference | `logs/inference/runs/<date>/<time>/csv/metrics.csv` |
| RQ-VAE / RK-Means training | `logs/train/runs/<date>/<time>/csv/metrics.csv` |
| RQ-VAE / RK-Means inference (SIDs) | `logs/inference/runs/<date>/<time>/csv/metrics.csv`, pickle at `.../pickle/merged_predictions_tensor.pt` |
| TIGER training | `logs/train/runs/<date>/<time>/csv/metrics.csv` |

The scripts report the exact CSV path after the corresponding stage finishes
(`train_tiger` prints it).  Checkpoints are saved under
`logs/train/runs/<date>/<time>/checkpoints/`.

## Shipped result logs (`results/`)

The four configurations of the dissertation are stored as exact copies of the
logged CSVs that produced the thesis tables:

| Config key (thesis name) | Directory | RQ-VAE log | TIGER log |
|---|---|---|---|
| `text_qwen3_0.6B` (Text 0.6B) | `results/text_qwen3_0.6B/` | `rqvae_metrics.csv` | `tiger_metrics.csv` |
| `text_qwen3_8B` (Text 8B) | `results/text_qwen3_8B/` | `rqvae_metrics.csv` | `tiger_metrics.csv` |
| `mmA_clip_text_concat` (Multimodal A) | `results/mmA_clip_text_concat/` | `rqvae_metrics.csv` | `tiger_metrics.csv` |
| `mmB_qwen3vl_2B` (Multimodal B) | `results/mmB_qwen3vl_2B/` | `rqvae_metrics.csv` | `tiger_metrics.csv` |

## RQ-VAE metrics columns

Each metric appears as three columns, e.g. `train/frac_unique_ids_step`,
`train/frac_unique_ids_epoch` and (for some) `train/frac_unique_ids`.  The
`*_step` variant is the value logged at the given `step`; use the **last row**
(`step = 3003`) as the final diagnostic, exactly as in the dissertation.

| Column | Meaning | Thesis use |
|---|---|---|
| `train/frac_unique_ids_step` | fraction of items whose 3-code Semantic ID is unique | "Unique IDs" (`1 - fraction` = inferred collision rate) |
| `train/layer_k/frac_layer_coverages_step` | codebook coverage of hierarchy level `k` (fraction of the 256 codes that are used) | "Cov. L0/L1/L2" |
| `train/layer_k/id_entropy_step` | entropy (bits) of the code distribution at level `k` (upper bound `log2 256 = 8`) | "Entropy L0/L1/L2" |
| `train/first_residuals_norm_ratio_step` | norm of the residual after the first codebook, relative to the input | "Residual ratio" |
| `train/last_residuals_norm_ratio_step` | norm of the residual after the last codebook | residual structure |
| `train/reconstruction_loss_step` | RQ-VAE reconstruction loss | "Recon. loss" |
| `train/mse_step`, `train/quantization_loss_step` | auxiliary loss terms (MSE, codebook/commitment) | – |
| `train/loss_step` | total training loss | – |

## TIGER metrics columns

| Column | Meaning |
|---|---|
| `train/loss_step`, `train/loss_epoch` | training loss |
| `val/recall@5`, `val/recall@10`, `val/ndcg@5`, `val/ndcg@10` | validation metrics (evaluated every `val_check_interval` steps) |
| `test/recall@5`, `test/recall@10`, `test/ndcg@5`, `test/ndcg@10` | test metrics on the leave-one-out test split (final row) |
| `test/loss` | test loss |

Validation/test values are only present on rows where the corresponding
evaluation ran; take the **final row** (`epoch = 1`) for the reported test
metrics.

## Expected values (dissertation results chapter)

Test-set metrics (`tiger_metrics.csv`, final row):

| Configuration | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | final step |
|---|---|---|---|---|---|
| Text 0.6B | 0.0262 | 0.0413 | 0.0172 | 0.0221 | 9100 |
| Text 8B | 0.0184 | 0.0284 | 0.0119 | 0.0152 | 7300 |
| Multimodal A | 0.0239 | 0.0371 | 0.0150 | 0.0193 | 6200 |
| Multimodal B | 0.0279 | 0.0411 | 0.0181 | 0.0224 | 7400 |

Semantic-ID diagnostics (`rqvae_metrics.csv`, final row, step 3003):

| Configuration | Unique IDs | Coverage L0/L1/L2 | Entropy L0/L1/L2 | Recon. loss | Residual ratio |
|---|---|---|---|---|---|
| Text 0.6B | 0.946 | 0.774/0.870/0.836 | 5.03/5.12/5.00 | 8.27e-4 | 5.5e-4 |
| Text 8B | 0.613 | 0.164/0.116/0.125 | 3.23/2.74/2.73 | 2.27e-4 | 1.6e-4 |
| Multimodal A | 0.934 | 0.478/0.707/0.677 | 4.47/4.82/4.71 | 5.72e-4 | 2.85e-2 |
| Multimodal B | 0.915 | 0.346/0.581/0.577 | 4.13/4.58/4.50 | 4.08e-4 | 2.60e-2 |

These are the values printed in the dissertation; a fresh run under the same
configuration is expected to land in the same range but will not reproduce the
exact digits (single seed, no averaging; see the README).
