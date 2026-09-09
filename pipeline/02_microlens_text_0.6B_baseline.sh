#!/usr/bin/env bash
# =============================================================================
# 02 - MicroLens main configuration: Text (Qwen3-Embedding-0.6B, title only)
#      (Experiment.ipynb, cell 2)
#
# The primary text-only baseline.  Pipeline:
#   1. data_process.py                MicroLens-50k -> TFRecords (leave-one-out)
#   2. sem_embeds_inference_flat      titles -> Qwen3-Embedding-0.6B (1024-d)
#   3. rqvae_train_flat               RQ-VAE, 3000 steps, 3 levels x 256 codes
#   4. rqvae_inference_flat           3-token Semantic IDs + GRID de-dup digit
#   5. tiger_train_flat               TIGER on 4-token IDs (1 epoch)
#
# Usage:
#   bash pipeline/02_microlens_text_0.6B_baseline.sh [all|data|embeddings|rqvae|tiger]
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_grid
install_configs

STATE_KEY="text_0.6B"
TEXT_EMBEDS="$(microlens_data_dir)/qwen3_text_0.6B_embeds.pt"

run_data() {
  # Command (original): python data_process.py
  microlens_data_process
}

run_embeddings() {
  # Command (original):
  #   python -m src.inference experiment=sem_embeds_inference_flat \
  #       data_dir=data/microLen \
  #       embedding_model="Qwen/Qwen3-Embedding-0.6B" \
  #       model.semantic_embedding_model.huggingface_model._target_="transformers.AutoModel.from_pretrained" \
  #       trainer.devices=1
  extract_qwen3_text_embeddings "$TEXT_EMBEDS" "Qwen/Qwen3-Embedding-0.6B" 1024
}

run_rqvae() {
  # Commands (original):
  #   python -m src.train experiment=rqvae_train_flat \
  #       data_dir=data/microLen embedding_path=<0.6B embeds> \
  #       embedding_dim=1024 num_hierarchies=3 codebook_width=256 trainer.devices=1
  #   python -m src.inference experiment=rqvae_inference_flat \
  #       data_dir=data/microLen embedding_path=<0.6B embeds> \
  #       embedding_dim=1024 num_hierarchies=3 codebook_width=256 \
  #       ckpt_path=<rqvae checkpoint step 3000>
  [ -f "$TEXT_EMBEDS" ] || die "text embeddings missing at $TEXT_EMBEDS (run the 'embeddings' step first)"
  train_rqvae "$TEXT_EMBEDS" 1024 "$STATE_KEY"
  infer_semantic_ids "$TEXT_EMBEDS" 1024 "$STATE_KEY"
}

run_tiger() {
  # Command (original):
  #   python -m src.train experiment=tiger_train_flat \
  #       data_dir=data/microLen semantic_id_path=<SID pickle> num_hierarchies=4
  train_tiger "$STATE_KEY"
}

case "${1:-all}" in
  all)        run_data; run_embeddings; run_rqvae; run_tiger ;;
  data)       run_data ;;
  embeddings) run_embeddings ;;
  rqvae)      run_rqvae ;;
  tiger)      run_tiger ;;
  *) die "unknown step '${1:-}' (use: all | data | embeddings | rqvae | tiger)" ;;
esac

info "Text-0.6B baseline finished."
