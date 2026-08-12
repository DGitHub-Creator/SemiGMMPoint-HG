#!/usr/bin/env bash
# Shared environment / path / logging helpers for the ODPT-HG SemiGMMPoint
# one-click entry scripts. Must be sourced from scripts/odpt_hg/*.
#
# ODPT-HG facts (audited):
#   dataset_name : ODPT-HG
#   dataset_root : /path/to/odpt-hg-dataset        (read-only raw txt)
#   data_root    : /path/to/odpt-hg-data      (converted .pth +
#                   splits + PROVENANCE.txt; can be rebuilt by
#                   tools/odpt_hg_convert.py)
#   output_root  : experiments/odpt_hg
set -euo pipefail

# ---- repo / env ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semigmm
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/semigmm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONWARNINGS=ignore
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

PYTHON="$HOME/miniconda3/envs/semigmm/bin/python"

# ---- run identity ----------------------------------------------------------
BUDGET="${BUDGET:-}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-100}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    RUN_ID="${RUN_ID:-smoke}"
    EPOCHS="${SMOKE_EPOCHS:-2}"
else
    RUN_ID="${RUN_ID:-official}"
fi

# ---- budget paths (new output root; experiments/odpt stays untouched) ------
EXP_ROOT="$REPO_ROOT/experiments/odpt_hg"
BUDGET_DIR="$EXP_ROOT/${BUDGET}pct"
TRAIN_DIR="$BUDGET_DIR/runs/$RUN_ID/train"
CKPT_DIR="$BUDGET_DIR/checkpoints/$RUN_ID"
EVAL_DIR="$BUDGET_DIR/eval/$RUN_ID"
LOG_DIR="$BUDGET_DIR/logs/$RUN_ID"
FINAL_CKPT="$CKPT_DIR/final.pth"

# ---- pretrain (shared by both budgets) ------------------------------------
PRETRAIN_DIR="$EXP_ROOT/pretrain"
PRETRAIN_CKPT="$PRETRAIN_DIR/checkpoints/final.pth"
PRE_GMM_CFG_YAML="$REPO_ROOT/cfgs/odpt_hg/pre_gmm_hg.yaml"

CFG_YAML="$REPO_ROOT/cfgs/odpt_hg/semi_gmm_hg_split${BUDGET}.yaml"
PROTO_TOOL="$REPO_ROOT/tools/odpt_hg_check_protocol.py"
EVAL_TOOL="$REPO_ROOT/tools/odpt_hg_eval.py"
COLLECT_TOOL="$REPO_ROOT/tools/odpt_hg_collect_metrics.py"
FINGERPRINT_TOOL="$REPO_ROOT/tools/odpt_hg_fingerprint.py"
CW_TOOL="$REPO_ROOT/tools/odpt_hg_class_weights.py"
CW_JSON="$EXP_ROOT/class_weights/split${BUDGET}.json"

HG_SPLIT_SRC="/path/to/odpt-hg-data/splits/${BUDGET}.txt"
if [ -n "$BUDGET" ] && [ -f "$HG_SPLIT_SRC" ]; then
    SPLIT_SHA="$(sha256sum "$HG_SPLIT_SRC" | cut -d' ' -f1)"
else
    SPLIT_SHA=""
fi

if [ -n "$BUDGET" ]; then
    BATCH_SIZE="$(grep -E '^batch_size:' "$CFG_YAML" | head -1 | awk '{print $2}')"
else
    BATCH_SIZE=""
fi

# ---- helpers ---------------------------------------------------------------
odpt_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

# run with tee; exit code of the command (not tee) propagates (set -o pipefail).
run_logged() {
    local log="$1"
    shift
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN: would run: $*"
        echo "DRY_RUN: log: $log"
        return 0
    fi
    "$@" 2>&1 | tee -a "$log"
}

print_run_header() {
    mkdir -p "$LOG_DIR"
    local log="$LOG_DIR/${1:-train}.log"
    {
        echo "================================================================"
        echo "SemiGMMPoint ODPT-HG ${BUDGET}% run header"
        odpt_log "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "dataset_name: ODPT-HG"
        echo "dataset_root: /path/to/odpt-hg-dataset"
        echo "data_root: /path/to/odpt-hg-data"
        if [ -f "$REPO_ROOT/experiments/odpt_hg/dataset_fingerprint.json" ]; then
            echo "dataset_fingerprint: $REPO_ROOT/experiments/odpt_hg/dataset_fingerprint.json"
        fi
        echo "split_fingerprint: $SPLIT_SHA ($HG_SPLIT_SRC)"
        if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            echo "git commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
        fi
        echo "python: $("$PYTHON" -c 'import sys; print(sys.version.split()[0])')"
        echo "pytorch: $("$PYTHON" -c 'import torch; print(torch.__version__)')"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/gpu: /'
        echo "config: $CFG_YAML"
        echo "labeled scenes: $(cat "$HG_SPLIT_SRC" | tr '\n' ' ')"
        echo "class weights artifact: $CW_JSON (from labeled-split GT only)"
        echo "test scenes: Area_3_conferenceRoom_20/21/22"
        echo "pretrained checkpoint (HG pre_gmm): $PRETRAIN_CKPT"
        echo "seed: $SEED"
        echo "batch size: $BATCH_SIZE"
        echo "epochs: $EPOCHS"
        echo "final checkpoint: $FINAL_CKPT"
        echo "ODPT-HG protocol: Area_3 evaluation disabled during training;"
        echo "final checkpoint evaluated only after training."
        echo "================================================================"
    } | tee -a "$log"
}

check_final_missing() {
    if [ ! -f "$FINAL_CKPT" ]; then
        die "找不到 final checkpoint：$FINAL_CKPT。请先运行训练脚本（如 bash scripts/odpt_hg/run_${BUDGET}_train.sh）。"
    fi
}

check_pretrain_missing() {
    if [ ! -f "$PRETRAIN_CKPT" ]; then
        die "找不到 HG pre_gmm checkpoint：$PRETRAIN_CKPT。请先运行 bash scripts/odpt_hg/run_pre_gmm.sh。"
    fi
}

guard_existing_output() {
    if [ -e "$FINAL_CKPT" ] && [ "$RESUME" != "1" ]; then
        die "已有输出存在：$FINAL_CKPT。如确需覆盖，请使用新的 RUN_ID（如 RUN_ID=run2 bash scripts/odpt_hg/run_${BUDGET}_train.sh），或显式设置 RESUME=1。"
    fi
}