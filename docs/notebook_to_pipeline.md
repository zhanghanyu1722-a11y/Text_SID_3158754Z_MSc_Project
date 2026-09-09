# From Experiment.ipynb to the pipeline scripts

The original experimental log is `Experiment.ipynb` (kept in the workspace
outside this package).  Each of its five cells is reproduced by exactly one
shell script in `pipeline/`.  Every GRID command of the notebook is preserved
verbatim inside the corresponding script as a comment directly above the
command that runs it; only the following generalisations were made:

1. **Output chaining.**  The notebook hard-codes the timestamped hydra output
   paths of the run that happened to precede each step (e.g.
   `logs/inference/runs/2026-07-29/08-06-54/pickle/merged_predictions_tensor.pt`).
   The scripts instead locate the output of the command they just executed
   (and remember it in `<GRID_ROOT>/.pipeline_state/`), so a fresh run chains
   automatically.  Frozen embeddings are additionally cached as canonical
   files under `data/microLen/*.pt` so stages can be re-run independently.
2. **Single-GPU default.**  `trainer.devices=1` (as in the MicroLens notebook
   cells) is applied uniformly through `NUM_DEVICES` (default 1); the original
   Amazon Beauty cell used all available GPUs.
3. **No checkpoint resume for the TIGER validation run.**  The original
   Amazon Beauty TIGER command resumed from an earlier checkpoint
   (`ckpt_path=...checkpoint_epoch=000_step=003100.ckpt`).  The validation
   script starts from scratch, since the step only serves as an end-to-end
   sanity check and no Beauty numbers are reported in the dissertation.
4. **Single-epoch protocol made explicit.**  The reported MicroLens runs were
   stopped after one epoch; the scripts pass `trainer.max_epochs=$TIGER_MAX_EPOCHS`
   (default 1) for reproducibility.

## Cell-to-script mapping

| Notebook cell | Purpose | Script |
|---|---|---|
| Cell 1 | Reproduce TIGER on Amazon Beauty (framework validation) | `pipeline/01_validate_amazon_beauty.sh` |
| Cell 2 | Qwen3-Embedding-0.6B text-only baseline | `pipeline/02_microlens_text_0.6B_baseline.sh` |
| Cell 3 | Multimodal A: CLIP ViT-B/32 + text concat (double-L2) | `pipeline/03_microlens_multimodal_A_clip.sh` |
| Cell 4 | Multimodal B: Qwen3-VL-Embedding-2B (unified) | `pipeline/04_microlens_multimodal_B_qwen3vl.sh` |
| Cell 5 | Text scaling: Qwen3-Embedding-8B (fp16) | `pipeline/05_microlens_text_8B.sh` |
| – | Run everything sequentially | `pipeline/run_all.sh` |

## Original notebook commands vs. script internals

**Cell 1 (Amazon Beauty):**

```bash
# extract flan-t5-xl item embeddings (2048-d)
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty
# residual-KMeans training
python -m src.train experiment=rkmeans_train_flat data_dir=data/amazon_data/beauty \
    embedding_path=<...>/merged_predictions_tensor.pt embedding_dim=2048 \
    num_hierarchies=3 codebook_width=256
# generate Semantic IDs
python -m src.inference experiment=rkmeans_inference_flat data_dir=data/amazon_data/beauty \
    embedding_path=<...>/merged_predictions_tensor.pt embedding_dim=2048 \
    num_hierarchies=3 codebook_width=256 ckpt_path=<rkmeans checkpoint>
# TIGER training (original resumed from an earlier checkpoint)
python -m src.train experiment=tiger_train_flat data_dir=data/amazon_data/beauty \
    semantic_id_path=<SID pickle> num_hierarchies=4
```

**Cell 2 (Text 0.6B baseline):**

```bash
python data_process.py
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/microLen \
    embedding_model="Qwen/Qwen3-Embedding-0.6B" \
    model.semantic_embedding_model.huggingface_model._target_="transformers.AutoModel.from_pretrained" \
    trainer.devices=1
python -m src.train experiment=rqvae_train_flat data_dir=data/microLen \
    embedding_path=<0.6B embeds> embedding_dim=1024 num_hierarchies=3 codebook_width=256 \
    trainer.devices=1
python -m src.inference experiment=rqvae_inference_flat data_dir=data/microLen \
    embedding_path=<0.6B embeds> embedding_dim=1024 num_hierarchies=3 codebook_width=256 \
    ckpt_path=<rqvae checkpoint step 3000>
python -m src.train experiment=tiger_train_flat data_dir=data/microLen \
    semantic_id_path=<SID pickle> num_hierarchies=4
```

**Cell 3 (Multimodal A):**

```bash
python extract_clip.py
python merge_features.py
python -m src.train experiment=rqvae_train_flat data_dir=data/microLen \
    embedding_path=data/microLen/multimodal_embeds.pt embedding_dim=1536 \
    num_hierarchies=3 codebook_width=256 trainer.devices=1
python -m src.inference experiment=rqvae_inference_flat data_dir=data/microLen \
    embedding_path=data/microLen/multimodal_embeds.pt embedding_dim=1536 \
    num_hierarchies=3 codebook_width=256 ckpt_path=<rqvae checkpoint step 3000>
python -m src.train experiment=tiger_train_flat data_dir=data/microLen \
    semantic_id_path=<SID pickle> num_hierarchies=4 trainer.devices=1
```

**Cell 4 (Multimodal B):**

```bash
python extract_vl_embedding.py
python -m src.train experiment=rqvae_train_flat data_dir=data/microLen \
    embedding_path=data/microLen/multimodal_vl_embeds.pt embedding_dim=2048 \
    num_hierarchies=3 codebook_width=256 trainer.devices=1
python -m src.inference experiment=rqvae_inference_flat data_dir=data/microLen \
    embedding_path=data/microLen/multimodal_vl_embeds.pt embedding_dim=2048 \
    num_hierarchies=3 codebook_width=256 ckpt_path=<rqvae checkpoint step 3000>
python -m src.train experiment=tiger_train_flat data_dir=data/microLen \
    semantic_id_path=<SID pickle> num_hierarchies=4 trainer.devices=1
```

**Cell 5 (Text 8B scaling):**

```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/microLen \
    embedding_model="Qwen/Qwen3-Embedding-8B" \
    model.semantic_embedding_model.huggingface_model._target_="transformers.AutoModel.from_pretrained" \
    +model.semantic_embedding_model.huggingface_model.dtype="float16" \
    trainer.devices=1
python -m src.train experiment=rqvae_train_flat data_dir=data/microLen \
    embedding_path=data/microLen/qwen8b_text_embeds_train.pt embedding_dim=4096 \
    num_hierarchies=3 codebook_width=256 trainer.devices=1
python -m src.inference experiment=rqvae_inference_flat data_dir=data/microLen \
    embedding_path=data/microLen/qwen8b_text_embeds_train.pt embedding_dim=4096 \
    num_hierarchies=3 codebook_width=256 ckpt_path=<rqvae checkpoint step 3000>
python -m src.train experiment=tiger_train_flat data_dir=data/microLen \
    semantic_id_path=<SID pickle> num_hierarchies=4
```

The scripts reproduce these commands while replacing the `<...>` placeholders
with automatically located artifacts (see `pipeline/common.sh`).
