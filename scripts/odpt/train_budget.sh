#!/usr/bin/env bash
# ODPT train entry: SemiGMMPoint <BUDGET>% (labeling only, val-aware pool).
# Usage:
#   bash scripts/odpt/run_10_train.sh            # official run (100 epochs)
#   SMOKE=1 bash scripts/odpt/run_10_train.sh    # smoke: 2 epochs, ~5 min
#   DRY_RUN=1 bash scripts/odpt/run_10_train.sh  # print commands only
#   RESUME=1 bash scripts/odpt/run_10_train.sh   # allow rerun over existing output
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "$TRAIN_DIR" "$CKPT_DIR" "$LOG_DIR"
print_run_header train
if [ "$DRY_RUN" != "1" ]; then
    guard_existing_output
fi

# ---- split file (canonical budget split for the labeling + val-aware pool) --
mkdir -p "$(dirname "$SPLIT_DST")"
run_logged "$LOG_DIR/train.log" cp "$SPLIT_SRC" "$SPLIT_DST"

# ---- protocol check (run before the experiment, mirrored in logs) ----------
run_logged "$LOG_DIR/protocol.log" "$PYTHON" "$PROTO_TOOL" --budget "$BUDGET"
run_logged "$LOG_DIR/train.log" "$PYTHON" "$PROTO_TOOL" --budget "$BUDGET"

# ---- train (labeling only, Area_3 evaluation disabled) ---------------------
# Always run from the repo root: the experiment dir is resolved relative to
# CWD, so running from scripts/odpt would pollute scripts/odpt/experiments/.
cd "$REPO_ROOT"
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
    [ -f "$RESUME_FROM" ] || die "resume_from checkpoint not found: $RESUME_FROM"
    RESUME_ARGS=(--resume_from "$RESUME_FROM")
    odpt_log "RESUMING from $RESUME_FROM"
fi
run_logged "$LOG_DIR/train.log" "$PYTHON" "$REPO_ROOT/examples/segmentation/semi_gmmpoint_main.py" \
    --cfg "$CFG_YAML" \
    --num_workers 4 \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --odpt_final_ckpt "$FINAL_CKPT" \
    --disable_validation True \
    --num_debug_gmm True \
    --tqdm_rank0_only True \
    --tqdm_ascii True \
    "${RESUME_ARGS[@]}"

# ---- fixed final checkpoint written by the protocol branch of the script ---
ls -la "$CKPT_DIR" | tee -a "$LOG_DIR/train.log"
if [ "$DRY_RUN" != "1" ]; then
    [ -f "$FINAL_CKPT" ] || die "final checkpoint not produced: $FINAL_CKPT"
fi

odpt_log "training done: final checkpoint = $FINAL_CKPT"
echo "TRAIN_OK"
