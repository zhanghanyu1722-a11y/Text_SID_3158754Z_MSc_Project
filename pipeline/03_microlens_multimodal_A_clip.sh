#!/usr/bin/env bash
# =============================================================================
# 03 - MicroLens main configuration: Multimodal A (CLIP ViT-B/32 + Qwen3-0.6B,
#      double-L2 concatenation)  -- Experiment.ipynb, cell 3
#
# Dual-tower (unaligned) multimodal family.  Pipeline:
#   1. extract_clip.py              covers -> CLIP ViT-B/32 (512-d, L2)
#   2. merge_features.py            double-L2 concat: text(1024) || image(512)
#                                   -> 1536-d fused vector
#   3. rqvae_train_flat / rqvae_inference_flat / tiger_train_flat
#
# Requires the text-0.6B embeddings produced by script 02 (they are reused as
# the text tower, exactly as in the original experiment).
#
# Usage:
#   bash pipeline/03_microlens_multimodal_A_clip.sh [all|embeddings|rqvae|tiger]
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_grid
install_configs

STATE_KEY="mmA_clip_concat"
TEXT_EMBEDS="$(microlens_data_dir)/qwen3_text_0.6B_embeds.pt"   # produced by 02
CLIP_EMBEDS="$(microlens_data_dir)/clip_image_embeds.pt"
FUSED_EMBEDS="$(microlens_data_dir)/mmA_text_clip_concat_embeds.pt"

run_embeddings() {
  # Commands (original):
  #   python extract_clip.py
  #   python merge_features.py
  [ -f "$TEXT_EMBEDS" ] \
    || die "text-0.6B embeddings missing at $TEXT_EMBEDS (run pipeline/02 first)"
  extract_clip_image_embeddings "$CLIP_EMBEDS"
  fuse_text_clip_double_l2 "$TEXT_EMBEDS" "$CLIP_EMBEDS" "$FUSED_EMBEDS"
}

run_rqvae() {
  # Commands (original):
  #   python -m src.train experiment=rqvae_train_flat \
  #       data_dir=data/microLen embedding_path=data/microLen/multimodal_embeds.pt \
  #       embedding_dim=1536 num_hierarchies=3 codebook_width=256 trainer.devices=1
  #   python -m src.inference experiment=rqvae_inference_flat \
  #       data_dir=data/microLen embedding_path=data/microLen/multimodal_embeds.pt \
  #       embedding_dim=1536 num_hierarchies=3 codebook_width=256 \
  #       ckpt_path=<rqvae checkpoint step 3000>
  [ -f "$FUSED_EMBEDS" ] \
    || die "fused multimodal embeddings missing at $FUSED_EMBEDS (run the 'embeddings' step first)"
  train_rqvae "$FUSED_EMBEDS" 1536 "$STATE_KEY"
  infer_semantic_ids "$FUSED_EMBEDS" 1536 "$STATE_KEY"
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

info "Multimodal A (CLIP concat) finished."
