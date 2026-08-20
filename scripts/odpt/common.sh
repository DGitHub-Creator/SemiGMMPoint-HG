#!/usr/bin/env bash
# Shared environment / path / logging helpers for the ODPT SemiGMMPoint
# one-click entry scripts. Must be sourced from scripts/odpt/*.
set -euo pipefail

# ---- repo / env ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semigmm
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/semigmm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# suppress TensorFlow (imported transitively by mmcv) and oneDNN noise
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
# deterministic UTF-8 locale / non-interactive stdout
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

PYTHON="$HOME/miniconda3/envs/semigmm/bin/python"

# ---- run identity ----------------------------------------------------------
BUDGET="${BUDGET:?set BUDGET (10|20) before sourcing common.sh}"
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

# ---- budget paths ----------------------------------------------------------
BUDGET_DIR="$REPO_ROOT/experiments/odpt/${BUDGET}pct"
TRAIN_DIR="$BUDGET_DIR/runs/$RUN_ID/train"
CKPT_DIR="$BUDGET_DIR/checkpoints/$RUN_ID"
EVAL_DIR="$BUDGET_DIR/eval/$RUN_ID"
LOG_DIR="$BUDGET_DIR/logs/$RUN_ID"
FINAL_CKPT="$CKPT_DIR/final.pth"

CFG_YAML="$REPO_ROOT/cfgs/odpt/semi_gmm_odpt_split${BUDGET}.yaml"
PROTO_TOOL="$REPO_ROOT/tools/odpt_check_protocol.py"
EVAL_TOOL="$REPO_ROOT/tools/odpt_eval.py"
COLLECT_TOOL="$REPO_ROOT/tools/odpt_collect_metrics.py"

SPLIT_SRC="/path/to/s3dis/data_split/${BUDGET}.txt"
SPLIT_DST="/path/to/odpt-data/splits/${BUDGET}.txt"

BATCH_SIZE="$(grep -E '^batch_size:' "$CFG_YAML" | head -1 | awk '{print $2}')"

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
        echo "SemiGMMPoint ODPT ${BUDGET}% run header"
        odpt_log "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
        if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            echo "git commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
            echo "git status:"
            git -C "$REPO_ROOT" status --short | sed 's/^/  /'
        else
            echo "git commit: not a git repository"
        fi
        echo "python: $("$PYTHON" -c 'import sys; print(sys.version.split()[0])')"
        echo "pytorch: $("$PYTHON" -c 'import torch; print(torch.__version__)')"
        echo "cuda: $("$PYTHON" -c 'import torch; print(torch.version.cuda)')"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/gpu: /'
        echo "config: $CFG_YAML"
        echo "split file: $SPLIT_DST (canonical: $SPLIT_SRC)"
        echo "labeled scenes: $(grep -E '^[0-9]' "$SPLIT_DST" 2>/dev/null || cat "$SPLIT_DST")"
        echo "unlabeled scenes: see protocol check output above"
        echo "test scenes: Area_3_conferenceRoom_20/21/22"
        echo "seed: $SEED"
        echo "batch size: $BATCH_SIZE"
        echo "epochs: $EPOCHS"
        echo "final checkpoint: $FINAL_CKPT"
        echo "ODPT protocol: Area_3 evaluation disabled during training."
        echo "ODPT protocol: final checkpoint will be evaluated only after training."
        echo "================================================================"
    } | tee -a "$log"
}

check_final_missing() {
    # eval: hard stop when the fixed final checkpoint does not exist.
    if [ ! -f "$FINAL_CKPT" ]; then
        die "找不到 final checkpoint：$FINAL_CKPT。请先运行训练脚本（如 bash scripts/odpt/run_${BUDGET}_train.sh）。"
    fi
}

guard_existing_output() {
    # train/eval must not silently overwrite previous full-run outputs.
    if [ -e "$FINAL_CKPT" ] && [ "$RESUME" != "1" ]; then
        die "已有输出存在：$FINAL_CKPT。如确需覆盖，请使用新的 RUN_ID（如 RUN_ID=run2 bash scripts/odpt/run_${BUDGET}_train.sh），或显式设置 RESUME=1。"
    fi
}
