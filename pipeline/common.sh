#!/usr/bin/env bash
# =============================================================================
# Common environment and helpers for the reproduction pipeline.
#
# Every pipeline script sources this file first.  All settings can be
# overridden through environment variables (see README section "Configuration").
#
#   PACKAGE_ROOT          root of this reproduction package (auto-detected)
#   GRID_ROOT             path to the snap-research/GRID checkout (REQUIRED)
#   MICROLENS_DATA_DIR    MicroLens data dir, relative to GRID_ROOT
#   MICROLENS_COVERS_DIR  cover-image dir, relative to GRID_ROOT
#   AMAZON_DATA_DIR       Amazon Beauty data dir, relative to GRID_ROOT
#   NUM_DEVICES           trainer.devices passed to every GRID command
#   TIGER_MAX_EPOCHS      max epochs for the TIGER training stage (default 1)
#   FORCE_EXTRACT         set to 1 to re-extract embeddings even if cached
#   PYTHON                python interpreter (default: python)
#
# Because every GRID hydra run writes to its own timestamped directory under
# <GRID_ROOT>/logs/..., pipeline stages remember the exact artifacts they
# produced in a small state directory (<GRID_ROOT>/.pipeline_state/<key>).
# This makes partial re-runs safe even when several configurations are run in
# the same checkout (the "newest log directory" heuristic alone would not be).
# =============================================================================
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ------------------------------------------------------------------ settings
PYTHON="${PYTHON:-python}"
GRID_ROOT="${GRID_ROOT:-}"
MICROLENS_DATA_DIR="${MICROLENS_DATA_DIR:-data/microLen}"
MICROLENS_COVERS_DIR="${MICROLENS_COVERS_DIR:-data/MicroLens-50k_covers}"
AMAZON_DATA_DIR="${AMAZON_DATA_DIR:-data/amazon_data/beauty}"
NUM_DEVICES="${NUM_DEVICES:-1}"
TIGER_MAX_EPOCHS="${TIGER_MAX_EPOCHS:-1}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

# --------------------------------------------------------------- tiny helpers
info() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(basename "$0")" "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

require_grid() {
  if [ -z "$GRID_ROOT" ]; then
    die "GRID_ROOT is not set. Export it, e.g.: export GRID_ROOT=\$HOME/code/GRID"
  fi
  [ -d "$GRID_ROOT/src" ] \
    || die "GRID_ROOT '$GRID_ROOT' does not look like a GRID checkout (missing src/)."
}

microlens_data_dir()   { printf '%s' "$GRID_ROOT/$MICROLENS_DATA_DIR"; }
microlens_covers_dir() { printf '%s' "$GRID_ROOT/$MICROLENS_COVERS_DIR"; }

# -------------------------------------------------------------- run helpers
# Every GRID hydra command must run with the GRID root as the working
# directory, because GRID resolves data_dir and its logs/ output layout
# relative to the current directory.
run_in_grid() { ( cd "$GRID_ROOT" && "$@" ); }

# Run one of the python stage scripts shipped in this package (also from the
# GRID root, so that relative default paths keep working).
run_python_stage() {
  local script="$PACKAGE_ROOT/$1"
  shift
  ( cd "$GRID_ROOT" && "$PYTHON" "$script" "$@" )
}

# --------------------------------------------------------------- state files
state_dir() { printf '%s/.pipeline_state' "$GRID_ROOT"; }

save_state() {  # $1 key, $2 value (a file path)
  mkdir -p "$(state_dir)"
  printf '%s' "$2" > "$(state_dir)/$1"
}

load_state() {  # $1 key; prints the stored path, or nothing
  local f
  f="$(state_dir)/$1"
  [ -f "$f" ] && cat "$f"
}

# ------------------------------------------------------------- output lookup
# Hydra run dirs look like <base>/YYYY-MM-DD/HH-MM-SS.  These helpers return
# the *newest* run dir/file below a base directory.  Prefer the state files
# above whenever the artifact must belong to a specific configuration.
latest_run_dir() {
  # $1: base directory, e.g. $GRID_ROOT/logs/train/runs
  [ -d "$1" ] || die "no run directory under '$1' (did the previous stage run?)"
  local d
  d="$(find "$1" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | cut -d' ' -f 2-)"
  [ -n "$d" ] || die "no run directory under '$1' (did the previous stage run?)"
  printf '%s' "$d"
}

latest_inference_pickle() {
  printf '%s/pickle/merged_predictions_tensor.pt' \
    "$(latest_run_dir "$GRID_ROOT/logs/inference/runs")"
}

latest_train_ckpt() {
  local d ckpt
  d="$(latest_run_dir "$GRID_ROOT/logs/train/runs")"
  ckpt="$(ls "$d"/checkpoints/*.ckpt 2>/dev/null | head -n 1 || true)"
  [ -n "$ckpt" ] || die "no *.ckpt found under '$d/checkpoints'"
  printf '%s' "$ckpt"
}

last_training_csv() {
  printf '%s/csv/metrics.csv' "$(latest_run_dir "$GRID_ROOT/logs/train/runs")"
}

# --------------------------------------------------------------- configs
# Copy the hydra experiment configs shipped in this package into the GRID
# checkout.  rqvae_inference_flat.yaml is not present in upstream GRID and is
# required for the Semantic-ID inference stage; the other three files are
# pinned copies of the configs used for the reported runs.
install_configs() {
  require_grid
  mkdir -p "$GRID_ROOT/configs/experiment"
  for cfg in "$PACKAGE_ROOT"/configs/*.yaml; do
    cp -f "$cfg" "$GRID_ROOT/configs/experiment/" \
      || die "failed to install '$cfg' into GRID configs/experiment/"
  done
  info "Experiment configs installed into $GRID_ROOT/configs/experiment/"
}

# =============================================================================
# Stage runners shared by the MicroLens experiment scripts.
#
# Conventions:
#   * "state keys" (e.g. text_0.6B) identify one experimental configuration.
#   * RQ-VAE checkpoints / Semantic-ID pickles are saved to the state dir so
#     that a later partial re-run uses the right artifacts.
# =============================================================================

# ----------------------------------------------------------- 1. preprocessing
microlens_data_process() {
  info "Stage: MicroLens-50k -> TFRecords (leave-one-out splits)"
  run_python_stage scripts/data_preparation/data_process.py \
    --pairs   "$(microlens_data_dir)/MicroLens-50k_pairs.tsv" \
    --titles  "$(microlens_data_dir)/MicroLens-50k_titles.csv" \
    --out-dir "$(microlens_data_dir)"
}

# --------------------------------------------------- 2. text embeddings (GRID)
# Encodes item titles with a Qwen3-Embedding model through GRID's
# semantic-embedding inference module and caches the resulting tensor.
#   $1: canonical cache file (absolute path)
#   $2: HF model id
#   $3: embedding dimension
#   $4: 'fp16' for half precision (8B model), empty otherwise
extract_qwen3_text_embeddings() {
  local cache="$1" model="$2" dim="$3" precision="${4:-}"
  if [ -f "$cache" ] && [ "$FORCE_EXTRACT" != "1" ]; then
    info "Reusing cached text embeddings: $cache"
    return 0
  fi
  info "Extracting text embeddings with $model (dim=$dim, fp16=${precision:-no})"
  if [ "$precision" = "fp16" ]; then
    run_in_grid "$PYTHON" -m src.inference experiment=sem_embeds_inference_flat \
      data_dir="$MICROLENS_DATA_DIR" \
      embedding_model="$model" \
      "model.semantic_embedding_model.huggingface_model._target_=transformers.AutoModel.from_pretrained" \
      "+model.semantic_embedding_model.huggingface_model.dtype=float16" \
      trainer.devices="$NUM_DEVICES"
  else
    run_in_grid "$PYTHON" -m src.inference experiment=sem_embeds_inference_flat \
      data_dir="$MICROLENS_DATA_DIR" \
      embedding_model="$model" \
      "model.semantic_embedding_model.huggingface_model._target_=transformers.AutoModel.from_pretrained" \
      trainer.devices="$NUM_DEVICES"
  fi
  info "Caching embeddings to $cache"
  mkdir -p "$(dirname "$cache")"
  cp -f "$(latest_inference_pickle)" "$cache"
}

# ---------------------------------------------------- 3. CLIP covers (script)
#   $1: canonical cache file (absolute path)
extract_clip_image_embeddings() {
  local cache="$1"
  if [ -f "$cache" ] && [ "$FORCE_EXTRACT" != "1" ]; then
    info "Reusing cached CLIP image embeddings: $cache"
    return 0
  fi
  info "Extracting CLIP ViT-B/32 cover-image embeddings"
  run_python_stage scripts/embedding_extraction/extract_clip.py \
    --covers-dir "$(microlens_covers_dir)" \
    --titles-csv "$(microlens_data_dir)/MicroLens-50k_titles.csv" \
    --output "$cache"
}

# ----------------------------------------------- 4. Qwen3-VL-Embedding (script)
#   $1: canonical cache file (absolute path)
extract_qwen3vl_embeddings() {
  local cache="$1"
  if [ -f "$cache" ] && [ "$FORCE_EXTRACT" != "1" ]; then
    info "Reusing cached Qwen3-VL-Embedding-2B features: $cache"
    return 0
  fi
  info "Extracting Qwen3-VL-Embedding-2B title+cover features"
  run_python_stage scripts/embedding_extraction/extract_vl_embedding.py \
    --titles-csv "$(microlens_data_dir)/MicroLens-50k_titles.csv" \
    --covers-dir "$(microlens_covers_dir)" \
    --output "$cache" \
    --progress-file "$(microlens_data_dir)/mmB_qwen3vl_2B_extract_progress.json"
}

# -------------------------------------------------------- 5. feature fusion
#   $1: text feature tensor (absolute)
#   $2: image feature tensor (absolute)
#   $3: fused output tensor (absolute)
fuse_text_clip_double_l2() {
  info "Fusing text + CLIP features (double L2 normalisation + concat)"
  run_python_stage scripts/feature_fusion/merge_features.py \
    --text-features "$1" \
    --image-features "$2" \
    --output "$3"
}

# ------------------------------------------------------------ 6. RQ-VAE train
#   $1: embedding tensor (absolute)
#   $2: embedding dimension
#   $3: state key (the produced checkpoint path is stored under it)
train_rqvae() {
  local ckpt
  info "RQ-VAE training (dim=$2, 3 levels x 256 codes, 3000 steps)"
  run_in_grid "$PYTHON" -m src.train experiment=rqvae_train_flat \
    data_dir="$MICROLENS_DATA_DIR" \
    embedding_path="$1" \
    embedding_dim="$2" \
    num_hierarchies=3 \
    codebook_width=256 \
    trainer.devices="$NUM_DEVICES"
  ckpt="$(latest_train_ckpt)"
  save_state "${3}_rqvae_ckpt" "$ckpt"
  info "RQ-VAE checkpoint: $ckpt"
}

# ------------------------------------------------------- 7. RQ-VAE -> SIDs
#   $1: embedding tensor (absolute)
#   $2: embedding dimension
#   $3: state key (checkpoint read from ${3}_rqvae_ckpt, SID pickle written to ${3}_semantic_ids)
infer_semantic_ids() {
  local state_key="$3" ckpt sid_pt
  ckpt="$(load_state "${state_key}_rqvae_ckpt")"
  [ -n "$ckpt" ] && [ -f "$ckpt" ] \
    || die "no RQ-VAE checkpoint for '$state_key' (run the 'rqvae' step first)"
  info "RQ-VAE inference -> Semantic IDs (ckpt: $ckpt)"
  run_in_grid "$PYTHON" -m src.inference experiment=rqvae_inference_flat \
    data_dir="$MICROLENS_DATA_DIR" \
    embedding_path="$1" \
    embedding_dim="$2" \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path="$ckpt" \
    trainer.devices="$NUM_DEVICES"
  sid_pt="$(latest_inference_pickle)"
  save_state "${state_key}_semantic_ids" "$sid_pt"
  info "Semantic IDs saved to: $sid_pt"
}

# ------------------------------------------------------------- 8. TIGER train
#   $1: state key (Semantic-ID pickle read from ${1}_semantic_ids)
train_tiger() {
  local state_key="$1" sid_pt
  sid_pt="$(load_state "${state_key}_semantic_ids")"
  [ -n "$sid_pt" ] && [ -f "$sid_pt" ] \
    || die "no Semantic-ID pickle for '$state_key' (run the 'rqvae' step first)"
  info "TIGER training on $sid_pt"
  info "num_hierarchies=4 (3 quantised codes + 1 GRID de-duplication digit), max_epochs=$TIGER_MAX_EPOCHS"
  run_in_grid "$PYTHON" -m src.train experiment=tiger_train_flat \
    data_dir="$MICROLENS_DATA_DIR" \
    semantic_id_path="$sid_pt" \
    num_hierarchies=4 \
    trainer.devices="$NUM_DEVICES" \
    trainer.max_epochs="$TIGER_MAX_EPOCHS"
  info "TIGER metrics CSV: $(last_training_csv)"
}

# =============================================================================
# Amazon Beauty validation stages (framework validation, rkmeans variant)
# =============================================================================

# Item text embeddings with the GRID default model (google/flan-t5-xl, 2048-d).
extract_amazon_sem_embeddings() {
  local cache="$1"
  if [ -f "$cache" ] && [ "$FORCE_EXTRACT" != "1" ]; then
    info "Reusing cached Amazon text embeddings: $cache"
    return 0
  fi
  info "Extracting Amazon Beauty text embeddings (google/flan-t5-xl, 2048-d)"
  run_in_grid "$PYTHON" -m src.inference experiment=sem_embeds_inference_flat \
    data_dir="$AMAZON_DATA_DIR" \
    trainer.devices="$NUM_DEVICES"
  info "Caching embeddings to $cache"
  mkdir -p "$(dirname "$cache")"
  cp -f "$(latest_inference_pickle)" "$cache"
}

# Residual-KMeans semantic IDs (the GRID 'rkmeans' variant used in the
# dissertation's framework validation on Amazon Beauty).
train_amazon_rkmeans() {
  local cache="$1"
  info "RK-Means training (dim=2048, 3 levels x 256 codes)"
  run_in_grid "$PYTHON" -m src.train experiment=rkmeans_train_flat \
    data_dir="$AMAZON_DATA_DIR" \
    embedding_path="$cache" \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    trainer.devices="$NUM_DEVICES"
  info "RK-Means checkpoint: $(latest_train_ckpt)"
}

infer_amazon_rkmeans_semantic_ids() {
  local cache="$1"
  info "RK-Means inference -> Semantic IDs"
  run_in_grid "$PYTHON" -m src.inference experiment=rkmeans_inference_flat \
    data_dir="$AMAZON_DATA_DIR" \
    embedding_path="$cache" \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path="$(latest_train_ckpt)" \
    trainer.devices="$NUM_DEVICES"
}

train_amazon_tiger() {
  info "TIGER training on Amazon Beauty Semantic IDs"
  info "num_hierarchies=4 (3 quantised codes + 1 GRID de-duplication digit), max_epochs=$TIGER_MAX_EPOCHS"
  run_in_grid "$PYTHON" -m src.train experiment=tiger_train_flat \
    data_dir="$AMAZON_DATA_DIR" \
    semantic_id_path="$(latest_inference_pickle)" \
    num_hierarchies=4 \
    trainer.devices="$NUM_DEVICES" \
    trainer.max_epochs="$TIGER_MAX_EPOCHS"
  info "TIGER metrics CSV: $(last_training_csv)"
}
