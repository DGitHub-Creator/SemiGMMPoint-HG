#!/usr/bin/env bash
# ODPT-HG train entry: SemiGMMPoint <BUDGET>% semi-supervised (labeling +
# unlabeled pool with y=255), loading the HG pre_gmm checkpoint.
# Usage:
#   bash scripts/odpt_hg/run_10_train.sh            # official (100 epochs)
#   SMOKE=1 bash scripts/odpt_hg/run_10_train.sh    # smoke: 2 epochs
#   DRY_RUN=1 bash scripts/odpt_hg/run_10_train.sh  # print commands only
#   RESUME=1 bash scripts/odpt_hg/run_10_train.sh   # allow rerun
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "$TRAIN_DIR" "$CKPT_DIR" "$LOG_DIR"
print_run_header train
if [ "$DRY_RUN" != "1" ]; then
    guard_existing_output
    check_pretrain_missing
fi

# ---- protocol check (before the experiment, mirrored in logs) -------------
run_logged "$LOG_DIR/protocol.log" "$PYTHON" "$PROTO_TOOL" --budget "$BUDGET"
run_logged "$LOG_DIR/train.log" "$PYTHON" "$PROTO_TOOL" --budget "$BUDGET"

# ---- per-split class weights artifact (labeled-split GT only) --------------
run_logged "$LOG_DIR/train.log" "$PYTHON" "$CW_TOOL" --budget "$BUDGET"

# ---- train (labeling only, validation disabled, Area_3 eval disabled) -----
cd "$REPO_ROOT"
run_logged "$LOG_DIR/train.log" "$PYTHON" "$REPO_ROOT/examples/segmentation/semi_gmmpoint_main.py" \
    --cfg "$CFG_YAML" \
    --num_workers 4 \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --pretrained_path "$PRETRAIN_CKPT" \
    --odpt_final_ckpt "$FINAL_CKPT" \
    --disable_validation True \
    --num_debug_gmm True \
    --tqdm_rank0_only True \
    --tqdm_ascii True

# ---- fixed final checkpoint written by the protocol branch of the script ---
ls -la "$CKPT_DIR" | tee -a "$LOG_DIR/train.log"
if [ "$DRY_RUN" != "1" ]; then
    [ -f "$FINAL_CKPT" ] || die "final checkpoint not produced: $FINAL_CKPT"
fi

odpt_log "training done: final checkpoint = $FINAL_CKPT"
echo "TRAIN_OK"