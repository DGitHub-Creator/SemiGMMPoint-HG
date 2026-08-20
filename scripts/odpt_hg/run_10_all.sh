#!/usr/bin/env bash
# SemiGMMPoint ODPT-HG 10%: train then standard Area_3 eval.
set -euo pipefail
export BUDGET=10
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$DIR/train_budget.sh" "$@"
"$DIR/eval_budget.sh" "$@"
