#!/usr/bin/env bash
# =============================================================================
# 01 - Framework validation on Amazon Beauty (Experiment.ipynb, cell 1)
#
# Reproduces the reference TIGER/GRID pipeline on the Amazon Beauty dataset to
# validate the codebase before the main MicroLens study.  This mirrors the
# GRID quick start: flan-t5-xl item embeddings -> residual-KMeans Semantic IDs
# (3 levels x 256 codes) -> TIGER training on 4-token IDs.
#
# NOTE: no recommendation numbers from this step are reported in the
# dissertation; it is an end-to-end sanity check of the implementation.
#
# Required data: P5-preprocessed Amazon Beauty TFRecords under
#   $GRID_ROOT/data/amazon_data/beauty/{items,training,evaluation,testing}
# (download + layout: see the GRID README; also README section "Data
# acquisition - Amazon Beauty").
#
# The three stages chain onto each other through the newest GRID log output,
# so run them in order (or as the full chain 'all'); do not interleave them
# with other configurations.
#
# Usage:
#   bash pipeline/01_validate_amazon_beauty.sh [all|embeddings|rkmeans|tiger]
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_grid
install_configs

AMAZON_CACHE="$GRID_ROOT/$AMAZON_DATA_DIR/flan_t5xl_text_embeds.pt"

run_embeddings() {
  # Command (original): python -m src.inference experiment=sem_embeds_inference_flat \
  #     data_dir=data/amazon_data/beauty
  extract_amazon_sem_embeddings "$AMAZON_CACHE"
}

run_rkmeans() {
  # Commands (original):
  #   python -m src.train experiment=rkmeans_train_flat \
  #       data_dir=data/amazon_data/beauty embedding_path=<embeds> \
  #       embedding_dim=2048 num_hierarchies=3 codebook_width=256
  #   python -m src.inference experiment=rkmeans_inference_flat \
  #       data_dir=data/amazon_data/beauty embedding_path=<embeds> \
  #       embedding_dim=2048 num_hierarchies=3 codebook_width=256 \
  #       ckpt_path=<rkmeans checkpoint>
  require_grid
  [ -f "$AMAZON_CACHE" ] || die "Amazon embeddings missing at $AMAZON_CACHE (run the 'embeddings' step first)"
  train_amazon_rkmeans "$AMAZON_CACHE"
  infer_amazon_rkmeans_semantic_ids "$AMAZON_CACHE"
}

run_tiger() {
  # Command (original):
  #   python -m src.train experiment=tiger_train_flat \
  #       data_dir=data/amazon_data/beauty \
  #       semantic_id_path=<rkmeans-inference SID pickle> num_hierarchies=4
  require_grid
  [ -f "$AMAZON_CACHE" ] || die "Amazon embeddings missing at $AMAZON_CACHE (run the 'embeddings' step first)"
  train_amazon_tiger
}

case "${1:-all}" in
  all)        run_embeddings; run_rkmeans; run_tiger ;;
  embeddings) run_embeddings ;;
  rkmeans)    run_rkmeans ;;
  tiger)      run_tiger ;;
  *) die "unknown step '${1:-}' (use: all | embeddings | rkmeans | tiger)" ;;
esac

info "Amazon Beauty validation finished."
