#!/usr/bin/env bash
# ODPT-HG pre_gmm unsupervised pretraining on ALL Area_1 scenes.
# Only coords/colors/features + augmented unlabeled views are used; no GT and
# no GT-derived class weights. Produces experiments/odpt_hg/pretrain/checkpoints/final.pth
# Usage:
#   bash scripts/odpt_hg/run_pre_gmm.sh            # full-run pretrain
#   SMOKE=1 bash scripts/odpt_hg/run_pre_gmm.sh    # smoke: 2 epochs
#   SMOKE=1 SMOKE_EPOCHS=1 bash ...                # 1 epoch smoke
#   DRY_RUN=1 bash scripts/odpt_hg/run_pre_gmm.sh  # print commands only
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG_DIR="$PRETRAIN_DIR/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
print_run_header pretrain

# ---- protocol preflight (pretrain pool = all 15 Area_1, no GT usage) ------
run_logged "$LOG_DIR/protocol.log" "$PYTHON" "$PROTO_TOOL" --budget 10

# ---- fingerprint (must exist before any training artifact) ----------------
if [ ! -f "$REPO_ROOT/experiments/odpt_hg/dataset_fingerprint.json" ]; then
    run_logged "$LOG_DIR/fingerprint.log" "$PYTHON" "$FINGERPRINT_TOOL"
fi

# ---- pretrain --------------------------------------------------------------
cd "$REPO_ROOT"
run_logged "$LOG_DIR/pretrain.log" "$PYTHON" "$REPO_ROOT/examples/segmentation/pre_gmm_main.py" \
    --cfg "$PRE_GMM_CFG_YAML" \
    --num_workers 4 \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --num_debug_gmm True \
    --tqdm_rank0_only True \
    --tqdm_ascii True

# ---- harvest best checkpoint -> fixed pretrain checkpoint ------------------
if [ "$DRY_RUN" != "1" ]; then
    SRC_CKPT="$(ls -t "$REPO_ROOT"/experiments/odpt_hg/odpt_hg/odpt_hg-train-pre_gmm_hg-*/checkpoint/*_ckpt_best.pth 2>/dev/null | head -1)"
    [ -n "$SRC_CKPT" ] || die "pre_gmm best checkpoint not produced"
    mkdir -p "$(dirname "$PRETRAIN_CKPT")"
    cp "$SRC_CKPT" "$PRETRAIN_CKPT"
    ls -la "$PRETRAIN_DIR/checkpoints" | tee -a "$LOG_DIR/pretrain.log"
fi

odpt_log "pretrain done: fixed checkpoint = $PRETRAIN_CKPT"
echo "PRETRAIN_OK"