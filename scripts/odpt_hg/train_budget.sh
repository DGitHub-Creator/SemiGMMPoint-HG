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

# Optional cfg override, e.g. CFG_NAME=semi_gmm_hg_split10_v2 bash .../run_10_train.sh
if [ -n "${CFG_NAME:-}" ]; then
    CFG_YAML="$REPO_ROOT/cfgs/odpt_hg/${CFG_NAME}.yaml"
fi

# PRETRAINED=skip trains from scratch (v3 big-model runs; no HG pre_gmm init).
if [ "${PRETRAINED:-load}" = "skip" ]; then
    PRETRAIN_ARGS=()
    PRETRAIN_MODE=scratch
else
    PRETRAIN_ARGS=(--pretrained_path "$PRETRAIN_CKPT")
    PRETRAIN_MODE=pretrained
fi

mkdir -p "$TRAIN_DIR" "$CKPT_DIR" "$LOG_DIR"
print_run_header train
if [ "$DRY_RUN" != "1" ]; then
    guard_existing_output
    if [ "$PRETRAIN_MODE" = "pretrained" ]; then
        check_pretrain_missing
    fi
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
    "${PRETRAIN_ARGS[@]}" \
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