# Artifacts reference — embedding & Semantic-ID tensors

All tensors below refer to the **MicroLens-50k** study set: **19,220 items**
(dense ids `0..19219`).  Every file was byte-archived from the author's run
outputs; `artifacts/MANIFEST.csv` records the SHA-256 and size of each file.

> **How the metadata was verified.** The `.pt` files are PyTorch zip
> containers; the storage/type/shape information below was read directly from
> the container headers and raw storage bytes (no torch required).

## 1. Embeddings (`artifacts/<config>/embedding/`)

| File | Dtype | Tensor (torch.load) | Storage rows | Verified on rows 0..63 |
|---|---|---|---|---|
| `text_qwen3_0.6B/embedding/qwen3_text_0.6B_embeds.pt` | float32 | `[19220, 1024]` | 19,220 | L2 norm ≈ 62–84 (not normalised) |
| `text_qwen3_8B/embedding/qwen3_text_8B_embeds.pt` | float32 | `[19220, 4096]` | 38,440* | L2 norm ≈ 79–121 (not normalised) |
| `mmA_clip_text_concat/embedding/clip_image_embeds.pt` | float32 | `[19220, 512]` | 19,220 | L2 norm ≈ 1.000 (normalised) |
| `mmA_clip_text_concat/embedding/mmA_text_clip_concat_embeds.pt` | float32 | `[19220, 1536]` | 19,220 | L2 norm = 1.000 (double-L2 concat) |
| `mmB_qwen3vl_2B/embedding/mmB_qwen3vl_2B_embeds.pt` | float32 | `[19220, 2048]` | 19,220 | L2 norm ≈ 0.998–1.004 (normalised) |

\* The 8B file's *underlying storage* contains a second, unused copy of the
tensor (a GRID merging artefact).  `torch.load` returns only the
`[19220, 4096]` view; the extra storage bytes are inert.

All embeddings are row-aligned with the dense item ids (`row i` = item `i`)
and contain no all-zero rows among the sampled rows (Qwen3-VL contains a small
number of exact-zero entries, consistent with its black-placeholder path).

## 2. Semantic IDs (`artifacts/<config>/semantic_id/`)

Each `semantic_ids.pt` loads as an int64 tensor of shape `[4, 38440]`
(strided view of a `38440 × 4` code matrix).  **Only the first 19,220 rows
belong to the 19,220 study items**; the remaining rows are a GRID merging
artefact (near-constant padding) and are not used by the pipeline.

Per-item row layout: `[code_level_0, code_level_1, code_level_2, dedup]`
where `code_level_k ∈ 0..255` is the codebook index at hierarchy level `k`
(codebook width 256) and `dedup` is GRID's per-item de-duplication digit
(0 for most items, >0 for colliding items), giving the 4-token Semantic IDs
that the TIGER stage consumes (`num_hierarchies=4`).

## 3. Provenance / mapping

> **Author-confirmed provenance.**  These are the exact intermediate files of
> the final runs logged in `Experiment.ipynb`: the `semantic_id/*.pt` pickles
> were passed directly as `semantic_id_path` to the reported TIGER training
> runs (whose metrics appear in `results/*/tiger_metrics.csv`), and the
> `embedding/*.pt` tensors are the caches consumed by the corresponding
> RQ-VAE runs.  The earlier "step-3000 checkpoint" statement therefore refers
> to these files verbatim.

| Archived file | Original location in the experiment | Produced by |
|---|---|---|
| `text_qwen3_0.6B/embedding/…` | GRID `logs/inference/…/pickle/merged_predictions_tensor.pt` (Qwen3-0.6B run) | `sem_embeds_inference_flat` (GRID) |
| `text_qwen3_8B/embedding/…` | `data/microLen/qwen8b_text_embeds_train.pt` | `sem_embeds_inference_flat` (GRID, fp16) |
| `mmA_clip_text_concat/embedding/clip_image_embeds.pt` | `data/microLen/clip_image_embeds.pt` | `scripts/embedding_extraction/extract_clip.py` |
| `mmA_clip_text_concat/embedding/mmA_text_clip_concat_embeds.pt` | `data/microLen/multimodal_embeds.pt` | `scripts/feature_fusion/merge_features.py` |
| `mmB_qwen3vl_2B/embedding/mmB_qwen3vl_2B_embeds.pt` | `data/microLen/multimodal_vl_embeds.pt` | `scripts/embedding_extraction/extract_vl_embedding.py` |
| every `semantic_id/semantic_ids.pt` | GRID `logs/inference/…/pickle/merged_predictions_tensor.pt` | `rqvae_inference_flat` (GRID), input to `tiger_train_flat` |

Original archive names (before this package normalised them):
`merged_predictions_tensor.pt` (text-0.6B / text-8B / CLIP / Qwen-VL semantic
ids), `qwen8b_text_embeds_train.pt` (8B embeddings),
`multimodal_embeds.pt` (CLIP fused), `multimodal_vl_embeds.pt` (Qwen-VL).
