#!/usr/bin/env bash
# =============================================================================
# 04 - MicroLens main configuration: Multimodal B (Qwen3-VL-Embedding-2B,
#      unified title+cover space)  -- Experiment.ipynb, cell 4
#
# Unified (aligned) multimodal family.  Pipeline:
#   1. extract_vl_embedding.py    cover + "Represent this video titled: <title>"
#                                 -> Qwen3-VL-Embedding-2B (2048-d, L2, resume-safe)
#   2. rqvae_train_flat / rqvae_inference_flat / tiger_train_flat
#
# Usage:
#   bash pipeline/04_microlens_multimodal_B_qwen3vl.sh [all|embeddings|rqvae|tiger]
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_grid
install_configs

STATE_KEY="mmB_qwen3vl_2B"
VL_EMBEDS="$(microlens_data_dir)/mmB_qwen3vl_2B_embeds.pt"

run_embeddings() {
  # Command (original): python extract_vl_embedding.py
  extract_qwen3vl_embeddings "$VL_EMBEDS"
}

run_rqvae() {
  # Commands (original):
  #   python -m src.train experiment=rqvae_train_flat \
  #       data_dir=data/microLen embedding_path=data/microLen/multimodal_vl_embeds.pt \
  #       embedding_dim=2048 num_hierarchies=3 codebook_width=256 trainer.devices=1
  #   python -m src.inference experiment=rqvae_inference_flat \
  #       data_dir=data/microLen embedding_path=data/microLen/multimodal_vl_embeds.pt \
  #       embedding_dim=2048 num_hierarchies=3 codebook_width=256 \
  #       ckpt_path=<rqvae checkpoint step 3000>
  [ -f "$VL_EMBEDS" ] \
    || die "Qwen3-VL embeddings missing at $VL_EMBEDS (run the 'embeddings' step first)"
  train_rqvae "$VL_EMBEDS" 2048 "$STATE_KEY"
  infer_semantic_ids "$VL_EMBEDS" 2048 "$STATE_KEY"
}

run_tiger() {
  # Command (original):
  #   python -m src.train experiment=tiger_train_flat \
  #       data_dir=data/microLen semantic_id_path=<SID pickle> num_hierarchies=4 trainer.devices=1
  train_tiger "$STATE_KEY"
}

case "${1:-all}" in
  all)        run_embeddings; run_rqvae; run_tiger ;;
  embeddings) run_embeddings ;;
  rqvae)      run_rqvae ;;
  tiger)      run_tiger ;;
  *) die "unknown step '${1:-}' (use: all | embeddings | rqvae | tiger)" ;;
esac

info "Multimodal B (Qwen3-VL) finished."
