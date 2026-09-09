# Text–Image Semantic IDs for Generative Recommendation — Reproducible Code

This package reproduces the experiments of the MSc dissertation
*"Text-Image Semantic IDs for Generative Recommendation"*: a controlled
benchmark that asks whether **text–image item embeddings produce better
Semantic IDs than text-only embeddings** for TIGER-style generative
recommendation, and whether **larger embedding models survive RQ-VAE
quantization**.

The experimental design holds everything downstream of the item embedding
fixed (data splits, Semantic-ID length, codebook budget, RQ-VAE and TIGER
training setups) and varies only the **frozen upstream item representation**
across four configurations:

| Config key | Dissertation name | Encoder (frozen) | Item input | Dim. | Fusion | Results in `results/` |
|---|---|---|---|---|---|---|
| `text_0.6B` | Text (0.6B) | `Qwen/Qwen3-Embedding-0.6B` | title | 1024 | – | `text_qwen3_0.6B/` |
| `text_8B` | Text (8B) | `Qwen/Qwen3-Embedding-8B` (fp16) | title | 4096 | – | `text_qwen3_8B/` |
| `mmA_clip_concat` | Multimodal A | CLIP ViT-B/32 + Qwen3-Embedding-0.6B | cover + title | 1536 | concat + double L2 norm | `mmA_clip_text_concat/` |
| `mmB_qwen3vl_2B` | Multimodal B | `Qwen/Qwen3-VL-Embedding-2B` | cover + title | 2048 | unified (aligned) space | `mmB_qwen3vl_2B/` |

All four run through the identical remaining pipeline: **RQ-VAE quantization
(3 levels × 256 codes, 3,000 steps) → Semantic IDs → TIGER training (one
epoch)**.  In addition, a validation run reproduces the reference TIGER/GRID
pipeline on Amazon Beauty (`pipeline/01_validate_amazon_beauty.sh`).

---

## 1. Repository layout

```
TextImage_SID_reproduction/
├── README.md                    this file (start here)
├── ACKNOWLEDGMENTS.md           provenance, citations, license notes
├── configs/                     hydra experiment configs (copy into GRID)
│   ├── sem_embeds_inference_flat.yaml
│   ├── rqvae_train_flat.yaml
│   ├── rqvae_inference_flat.yaml     # added by the author (not in upstream GRID)
│   └── tiger_train_flat.yaml
├── scripts/                     python stage scripts (cleaned, parameterised)
│   ├── data_preparation/data_process.py        # MicroLens-50k -> TFRecords
│   ├── embedding_extraction/extract_clip.py    # CLIP ViT-B/32 cover features
│   ├── embedding_extraction/extract_vl_embedding.py  # Qwen3-VL-Embedding-2B
│   └── feature_fusion/merge_features.py        # double-L2 text+image concat
├── pipeline/                    end-to-end bash pipeline (one script per
│   ├── common.sh                    notebook cell; see docs/notebook_to_pipeline.md)
│   ├── 01_validate_amazon_beauty.sh        # Amazon Beauty validation
│   ├── 02_microlens_text_0.6B_baseline.sh  # Text (0.6B)
│   ├── 03_microlens_multimodal_A_clip.sh   # Multimodal A
│   ├── 04_microlens_multimodal_B_qwen3vl.sh# Multimodal B
│   ├── 05_microlens_text_8B.sh             # Text (8B)
│   └── run_all.sh                          # run 01-05 sequentially
├── results/                     byte-exact logged CSVs of the reported runs
│   ├── text_qwen3_0.6B/         rqvae_metrics.csv, tiger_metrics.csv
│   ├── text_qwen3_8B/
│   ├── mmA_clip_text_concat/
│   └── mmB_qwen3vl_2B/
├── artifacts/                   intermediate artifacts of the reported runs
│   │                            (frozen embeddings + Semantic-ID pickles;
│   │                            see docs/artifacts_reference.md)
│   ├── MANIFEST.csv             sha256 + size of every tensor below
│   ├── text_qwen3_0.6B/         embedding/ + semantic_id/
│   ├── text_qwen3_8B/
│   ├── mmA_clip_text_concat/
│   └── mmB_qwen3vl_2B/
└── docs/
    ├── metrics_reference.md     every CSV column + expected values (thesis tables)
    ├── artifacts_reference.md   every archived tensor: role, shape, provenance
    └── notebook_to_pipeline.md  mapping of the original Experiment.ipynb cells
```

The codebase that actually trains RQ-VAE and TIGER is the **GRID** repository
by Snap Research (an open implementation of TIGER with residual-quantization
Semantic IDs), referenced below — this package configures and drives it, and
adds the MicroLens-specific stages around it.

---

## 2. Environment setup

Target: **Linux with a CUDA GPU** (also works under WSL2 on Windows).
Python ≥ 3.10 is required by GRID.  The reported runs used a single GPU
(`trainer.devices=1`); the 8B text extraction additionally runs in float16.

```bash
# 1. GRID (the training/inference codebase) ----------------------------
git clone https://github.com/snap-research/GRID.git
cd GRID
git checkout 2fe3475b2d369580234093f35d52b1a2f54d0472   # pinned commit used here
pip install -r requirements.txt

# 2. extra packages needed by the multimodal stages ---------------------
pip install git+https://github.com/openai/CLIP.git      # extract_clip.py
pip install sentence-transformers                       # extract_vl_embedding.py
```

Notes

* The GRID `requirements.txt` already provides PyTorch, Lightning, Hydra,
  TensorFlow (CPU, used only to write/read TFRecords), transformers, etc.
* `transformers` may need to be newer than the version pinned by GRID to load
  the Qwen3-Embedding models natively.  If loading fails, upgrade
  `transformers` (and keep `pytorch-lightning`/`lightning` compatible, see the
  GRID README).
* HF authentication: models are public on the HuggingFace hub; set
  `HF_HOME`/`HUGGINGFACE_HUB_CACHE` as usual if you use a custom cache.

---

## 3. Data acquisition

**This package does not ship raw dataset files** (MicroLens terms of use
prohibit secondary redistribution; see `ACKNOWLEDGMENTS.md`).  The derived
intermediate tensors of the reported runs *are* included under
[`artifacts/`](artifacts/README.md) — frozen item embeddings and Semantic-ID
pickles — so the Semantic-ID structure can be inspected and the TIGER stage
re-run without re-extracting embeddings.

### 3.1 Amazon Beauty (validation only)

Download the P5-preprocessed Amazon data from the link in the [GRID
README](https://github.com/snap-research/GRID)
([Google Drive](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing))
and unpack it so that the Beauty split is visible as

```
<GRID_ROOT>/data/amazon_data/beauty/
├── items/            items.tfrecord.gz (id + text)
├── training/         train.tfrecord.gz
├── evaluation/       valid.tfrecord.gz
└── testing/          test.tfrecord.gz
```

(follow the GRID quick start for the exact archive layout; adjust
`AMAZON_DATA_DIR` if needed).

### 3.2 MicroLens-50k (main study)

The dissertation works on a 50,000-user subset of the official **MicroLens**
micro-video dataset: **50,000 users · 19,220 videos · 359,708 interactions**,
every item having exactly one title and one cover image.

1. Download the official MicroLens-50K data from
   https://recsys.westlake.edu.cn/ (dataset page; interactions, item
   text/captions, cover images).  Project page:
   https://github.com/westlake-repl/MicroLens.
2. Build the three files below from the official downloads (the subsetting
   step used by the author is outside this package, but the required content
   is fully specified here so any 50k-user MicroLens sample with titles and
   covers works).

Create this layout under your GRID clone:

```
<GRID_ROOT>/data/microLen/
├── MicroLens-50k_pairs.tsv       per user:  user_id <TAB> item_id item_id ... (chronological)
├── MicroLens-50k_titles.csv      columns:  item,title
└── (items/, training/, evaluation/, testing/  are created by data_process.py)

<GRID_ROOT>/data/MicroLens-50k_covers/
└── <original_item_id>.jpg        cover image per item (jpg)
```

File formats:

| File | Format | Example row |
|---|---|---|
| `MicroLens-50k_pairs.tsv` | one user per row; space-separated item ids already in chronological order (as in the official `*_pairs.tsv`) | `10476<TAB>20314 45389 1023 ...` |
| `MicroLens-50k_titles.csv` | header `item,title` | `20314,My awesome short video` |
| `MicroLens-50k_covers/<id>.jpg` | cover named by the *original* (pre-remap) item id | `20314.jpg` |

Sanity checks: 50,000 users, 19,220 titles rows, ~359,708 interactions,
every user with ≥ 5 interactions (the < 3 safeguard in `data_process.py`
never triggers).

> **Note.** `data_process.py` assumes each row of the pairs file is already
> sorted by interaction time (true for the official files).  Item ids are then
> remapped to a dense `0..N-1` range (sorted by original id) because GRID
> indexes items by dense integers.

---

## 4. Configuration

All pipeline scripts read environment variables (defaults in parentheses):

| Variable | Meaning |
|---|---|
| `GRID_ROOT` | **(required)** path to the snap-research/GRID checkout |
| `MICROLENS_DATA_DIR` | MicroLens data dir, relative to GRID root (`data/microLen`) |
| `MICROLENS_COVERS_DIR` | cover dir, relative to GRID root (`data/MicroLens-50k_covers`) |
| `AMAZON_DATA_DIR` | Amazon Beauty data dir (`data/amazon_data/beauty`) |
| `NUM_DEVICES` | `trainer.devices` for every GRID command (`1`) |
| `TIGER_MAX_EPOCHS` | epochs of TIGER training (`1`; single-epoch protocol of the dissertation) |
| `FORCE_EXTRACT` | `1` re-extracts embeddings even if the cache file exists (`0`) |
| `PYTHON` | python interpreter (`python`) |

Example:

```bash
export GRID_ROOT="$HOME/code/GRID"     # adapt
export NUM_DEVICES=1
```

---

## 5. Usage

Each configuration script runs its full chain by default and can also run a
single stage (useful after an interruption):

```bash
bash pipeline/01_validate_amazon_beauty.sh [all|embeddings|rkmeans|tiger]
bash pipeline/02_microlens_text_0.6B_baseline.sh [all|data|embeddings|rqvae|tiger]
bash pipeline/03_microlens_multimodal_A_clip.sh  [all|embeddings|rqvae|tiger]
bash pipeline/04_microlens_multimodal_B_qwen3vl.sh[all|embeddings|rqvae|tiger]
bash pipeline/05_microlens_text_8B.sh            [all|embeddings|rqvae|tiger]
```

### Recommended order

```bash
# 0. one-time: point the package at GRID and let it install the configs
export GRID_ROOT="$HOME/code/GRID"
bash pipeline/02_microlens_text_0.6B_baseline.sh data   # build the TFRecords once
```

Then the four configurations **in order** (03 reuses the 0.6B text
embeddings from 02):

```bash
bash pipeline/02_microlens_text_0.6B_baseline.sh          # Text 0.6B
bash pipeline/03_microlens_multimodal_A_clip.sh           # Multimodal A
bash pipeline/04_microlens_multimodal_B_qwen3vl.sh        # Multimodal B
bash pipeline/05_microlens_text_8B.sh                     # Text 8B
```

or everything at once with `bash pipeline/run_all.sh`
(`--with-validation` prepends the Amazon Beauty step).

Optional framework validation first:

```bash
bash pipeline/01_validate_amazon_beauty.sh                # needs the Beauty data
```

What happens under the hood (all original commands are preserved in the
script comments and in `docs/notebook_to_pipeline.md`):

| Script | Stage | Produces |
|---|---|---|
| `02/03/04/05` step `data` | `data_process.py` | TFRecords under `data/microLen/{items,training,evaluation,testing}` |
| step `embeddings` | GRID `sem_embeds_inference_flat` or the extraction scripts | cached `.pt` embedding tensors under `data/microLen/` |
| step `rqvae` | GRID `rqvae_train_flat` + `rqvae_inference_flat` | RQ-VAE checkpoint + Semantic-ID pickle |
| step `tiger` | GRID `tiger_train_flat` | TIGER metrics CSV |

Cached embedding files (`FORCE_EXTRACT=1` to recompute):

| File | Content |
|---|---|
| `data/microLen/qwen3_text_0.6B_embeds.pt` | 1024-d text embeddings (02) |
| `data/microLen/qwen3_text_8B_embeds.pt` | 4096-d text embeddings, fp16 (05) |
| `data/microLen/clip_image_embeds.pt` | 512-d CLIP cover features (03) |
| `data/microLen/mmA_text_clip_concat_embeds.pt` | 1536-d fused vector (03) |
| `data/microLen/mmB_qwen3vl_2B_embeds.pt` | 2048-d unified features (04) |

`data/microLen/mmB_qwen3vl_2B_extract_progress.json` is the resume state of
the Qwen3-VL extraction (safe to delete together with the `.pt` to restart).

> The exact frozen tensors and Semantic-ID pickles of the reported runs are
> archived in this package under `artifacts/` with the same canonical file
> names (one directory per configuration; see
> [`docs/artifacts_reference.md`](docs/artifacts_reference.md)).  The
> `semantic_id/semantic_ids.pt` pickles are byte-identical to the files passed
> as `semantic_id_path` to the TIGER training runs reported in the
> dissertation.  If you copy them to the `<GRID_ROOT>/data/microLen/` paths in
> the table above, the extraction stages of the pipeline skip straight to
> RQ-VAE / TIGER.

### Partial re-runs and state

Each configuration stores the exact artifacts it produced (RQ-VAE checkpoint,
Semantic-ID pickle) in `<GRID_ROOT>/.pipeline_state/<config>_*`.  Running
`bash pipeline/03_...sh rqvae` therefore re-uses the right files even if other
configurations were trained in the same checkout.

---

## 6. Outputs and how to read them

See [`docs/metrics_reference.md`](docs/metrics_reference.md) for the full
column reference and the expected numbers.  In short:

* **RQ-VAE run** → `logs/train/runs/<date>/<time>/csv/metrics.csv`:
  `train/frac_unique_ids_step`, `train/layer_{0,1,2}/frac_layer_coverages_step`,
  `train/layer_{0,1,2}/id_entropy_step`, `train/first_residuals_norm_ratio_step`,
  `train/reconstruction_loss_step`, ... Use the last row (`step=3003`) for the
  Semantic-ID diagnostics reported in the dissertation.
* **TIGER run** → `logs/train/runs/<date>/<time>/csv/metrics.csv`:
  `val/*` and `test/recall@5|10`, `test/ndcg@5|10`; use the last row for the
  reported test metrics.
* `results/` contains the byte-exact CSVs of the runs that produced the
  dissertation tables — compare a fresh run against them.

---

## 7. Expected results

The dissertation results (test set, TIGER after one epoch):

| Configuration | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| Text 0.6B | 0.0262 | 0.0413 | 0.0172 | 0.0221 |
| Text 8B | 0.0184 | 0.0284 | 0.0119 | 0.0152 |
| Multimodal A (CLIP concat) | 0.0239 | 0.0371 | 0.0150 | 0.0193 |
| Multimodal B (Qwen3-VL) | 0.0279 | 0.0411 | 0.0181 | 0.0224 |

Headline findings: multimodal embeddings are *not* automatically better —
unaligned CLIP+text concatenation hurts (Multimodal A below the text
baseline), while the aligned unified space of Qwen3-VL (Multimodal B) is best
overall; scaling the text encoder 0.6B → 8B **collapses** the quantized
codebook (unique-ID fraction 0.946 → 0.613) and cuts Recall@10 / NDCG@10 by
≈ 31 % even though reconstruction error improves.  See
`docs/metrics_reference.md` for the diagnostics table.

Every run was executed **once** (no seed averaging, no significance testing),
so a fresh run is expected to land in the same range without reproducing the
exact digits.  The RQ-VAE collapse of the 8B configuration is robust;
small-metric rankings are the stable claims.

---

## 8. Implementation notes and deviations (read before reporting numbers)

* **Qwen3-Embedding usage.** Text embeddings are produced through GRID's
  semantic-embedding inference module by overriding its HF model target to
  `transformers.AutoModel.from_pretrained` and mean-pooling the last hidden
  state (GRID's default aggregation).  This is *not* the official
  Qwen3-Embedding API (which recommends last-token pooling and task
  instructions); the same non-official recipe is used for both the 0.6B and
  8B runs so the comparison stays internal.
* **Qwen3-VL-Embedding usage.** Titles + covers are encoded with
  `sentence-transformers` (`trust_remote_code=True`) using the query format
  `Represent this video titled: <title>` and output L2 normalisation, as
  recommended by the model card.
* **8B extraction** runs in float16 to fit GPU memory.
* **Missing covers** are replaced by black placeholders (640×360) in the
  Qwen3-VL extraction and skipped-to-zero rows in CLIP extraction; neither
  path triggers on MicroLens-50k.
* **TIGER `num_hierarchies=4`.** GRID appends one extra token per item to
  de-duplicate colliding Semantic IDs, so the 4-token IDs fed to TIGER are the
  3 RQ-VAE codes plus that GRID token (see the GRID README).
* **Single epoch / single seed.** RQ-VAE trains 3,000 steps for every
  configuration; TIGER trains one epoch (`TIGER_MAX_EPOCHS=1`).  The original
  run-level step counts differed slightly per configuration (6,200–9,100
  gradient updates) — a caveat acknowledged in the dissertation.
* **Amazon validation** uses GRID's `rkmeans_*` (residual-KMeans) configs and
  the GRID default `google/flan-t5-xl` encoder; no Beauty metrics are reported
  in the dissertation.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `GRID_ROOT is not set` | `export GRID_ROOT=/path/to/GRID` (must contain `src/`) |
| Out of GPU memory during extraction | lower `--batch-size` of the extraction scripts (they are also invocable directly with `--help`) |
| Out of GPU memory during RQ-VAE/TIGER | append GRID overrides, e.g. `trainer.accumulate_grad_batches=32`, or reduce `data_loading.datamodule.train_dataloader_config.batch_size_per_device` on the GRID command line |
| Qwen3-Embedding fails to load | upgrade `transformers` to a release with native Qwen3 support |
| Qwen3-VL-Embedding download slow/fails | pre-download the model with `huggingface-cli download Qwen/Qwen3-VL-Embedding-2B` |
| Extraction interrupted | Qwen3-VL extraction resumes automatically; CLIP/text extraction restart and overwrite the same deterministic cache |
| Wrong artifacts after partial re-runs | run the full chain of a configuration (`...sh all`) or clean `<GRID_ROOT>/.pipeline_state/` |
| You are on Windows | run everything under WSL2 (the pipeline is bash); the original experiments were run the same way |

---

## 10. Further reading

* [`docs/notebook_to_pipeline.md`](docs/notebook_to_pipeline.md) — mapping to
  the original `Experiment.ipynb` commands.
* [`docs/metrics_reference.md`](docs/metrics_reference.md) — CSV columns and
  expected values.
* [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) — citations and provenance.
* Related: TIGER (arXiv:2305.05065), GRID (arXiv:2507.22224,
  https://github.com/snap-research/GRID), MicroLens (arXiv:2309.15379).
