#!/usr/bin/env bash
# =============================================================================
# Run the full reproduction in one go.
#
#   bash pipeline/run_all.sh            # main MicroLens experiments (02-05)
#   bash pipeline/run_all.sh --with-validation   # also Amazon Beauty (01)
#
# Each experiment script runs all of its own stages (data -> embeddings ->
# RQ-VAE -> TIGER).  Already-cached frozen embeddings are reused unless
# FORCE_EXTRACT=1 is exported.  GRID_ROOT must be set (see common.sh).
#
# CAUTION: this is the complete training schedule of the dissertation
# (four RQ-VAE runs of 3000 steps + four TIGER runs of one epoch, plus the
# 8B and 2B embedding extractions).  On a single GPU expect it to take a
# long time; run the numbered scripts individually to control the schedule.
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_grid

WITH_VALIDATION=0
[ "${1:-}" = "--with-validation" ] && WITH_VALIDATION=1

run_script() {
  local name="$1" start end
  start="$(date +%s)"
  info ">>> Running $name"
  bash "$(dirname "${BASH_SOURCE[0]}")/$name"
  end="$(date +%s)"
  info "<<< $name finished in $((end - start)) s"
}

info "GRID_ROOT=$GRID_ROOT"
info "NUM_DEVICES=$NUM_DEVICES  TIGER_MAX_EPOCHS=$TIGER_MAX_EPOCHS  FORCE_EXTRACT=$FORCE_EXTRACT"

if [ "$WITH_VALIDATION" = "1" ]; then
  run_script 01_validate_amazon_beauty.sh
fi
run_script 02_microlens_text_0.6B_baseline.sh
run_script 03_microlens_multimodal_A_clip.sh
run_script 04_microlens_multimodal_B_qwen3vl.sh
run_script 05_microlens_text_8B.sh

info "All runs finished."
