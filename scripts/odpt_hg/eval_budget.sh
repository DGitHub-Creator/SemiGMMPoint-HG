#!/usr/bin/env bash
# ODPT-HG eval entry: evaluate the FINAL checkpoint on exactly the 3 Area_3
# scenes, then refresh experiments/odpt_hg/summary.*
# Usage:
#   bash scripts/odpt_hg/run_10_eval.sh             # official eval (after train)
#   SMOKE=1 bash scripts/odpt_hg/run_10_eval.sh     # smoke: allow non-final ckpt
#   DRY_RUN=1 bash scripts/odpt_hg/run_10_eval.sh   # print commands only
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "$EVAL_DIR" "$LOG_DIR"
print_run_header eval

# ---- hard stop when the fixed final checkpoint is missing ------------------
if [ "$DRY_RUN" != "1" ]; then
    check_final_missing
fi

# ---- protocol check before evaluation --------------------------------------
run_logged "$LOG_DIR/eval.log" "$PYTHON" "$PROTO_TOOL" --budget "$BUDGET"

# ---- official evaluation on the final checkpoint ---------------------------
cd "$REPO_ROOT"
EVAL_ARGS=(--budget "$BUDGET" --cfg "$CFG_YAML" --checkpoint "$FINAL_CKPT"
           --outdir "$EVAL_DIR" --epochs "$EPOCHS")
if [ "$SMOKE" = "1" ]; then
    EVAL_ARGS+=(--smoke)
fi
run_logged "$LOG_DIR/eval.log" "$PYTHON" "$EVAL_TOOL" "${EVAL_ARGS[@]}"

# ---- refresh the summary ---------------------------------------------------
run_logged "$LOG_DIR/eval.log" "$PYTHON" "$COLLECT_TOOL" --run-id "$RUN_ID"
echo "EVAL_OK"
