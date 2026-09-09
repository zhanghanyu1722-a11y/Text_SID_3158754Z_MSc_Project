#!/usr/bin/env bash
# =============================================================================
# 05 - MicroLens main configuration: Text scaling (Qwen3-Embedding-8B, fp16)
#      -- Experiment.ipynb, cell 5
#
# Model-scaling arm of the text-only family.  Identical pipeline to 02 but with
# Qwen3-Embedding-8B (4096-d, loaded in float16 to fit in GPU memory):
#   1. sem_embeds_inference_flat      titles -> Qwen3-Embedding-8B (4096-d, fp16)
#   2. rqvae_train_flat / rqvae_inference_flat / tiger_train_flat
#
# Usage:
#   bash pipeline/05_microlens_text_8B.sh [all|embeddings|rqvae|tiger]
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_grid
install_configs

STATE_KEY="text_8B"
TEXT_EMBEDS="$(microlens_data_dir)/qwen3_text_8B_embeds.pt"

run_embeddings() {
  # Command (original):
  #   python -m src.inference experiment=sem_embeds_inference_flat \
  #       data_dir=data/microLen \
  #       embedding_model="Qwen/Qwen3-Embedding-8B" \
  #       model.semantic_embedding_model.huggingface_model._target_="transformers.AutoModel.from_pretrained" \
  #       +model.semantic_embedding_model.huggingface_model.dtype="float16" \
  #       trainer.devices=1
  extract_qwen3_text_embeddings "$TEXT_EMBEDS" "Qwen/Qwen3-Embedding-8B" 4096 fp16
}

run_rqvae() {
  # Commands (original):
  #   python -m src.train experiment=rqvae_train_flat \
  #       data_dir=data/microLen embedding_path=data/microLen/qwen8b_text_embeds_train.pt \
  #       embedding_dim=4096 num_hierarchies=3 codebook_width=256 trainer.devices=1
  #   python -m src.inference experiment=rqvae_inference_flat \
  #       data_dir=data/microLen embedding_path=data/microLen/qwen8b_text_embeds_train.pt \
  #       embedding_dim=4096 num_hierarchies=3 codebook_width=256 \
  #       ckpt_path=<rqvae checkpoint step 3000>
  [ -f "$TEXT_EMBEDS" ] \
    || die "text-8B embeddings missing at $TEXT_EMBEDS (run the 'embeddings' step first)"
  train_rqvae "$TEXT_EMBEDS" 4096 "$STATE_KEY"
  infer_semantic_ids "$TEXT_EMBEDS" 4096 "$STATE_KEY"
}

run_tiger() {
  # Command (original):
  #   python -m src.train experiment=tiger_train_flat \
  #       data_dir=data/microLen semantic_id_path=<SID pickle> num_hierarchies=4
  train_tiger "$STATE_KEY"
}

case "${1:-all}" in
  all)        run_embeddings; run_rqvae; run_tiger ;;
  embeddings) run_embeddings ;;
  rqvae)      run_rqvae ;;
  tiger)      run_tiger ;;
  *) die "unknown step '${1:-}' (use: all | embeddings | rqvae | tiger)" ;;
esac

info "Text-8B scaling run finished."
