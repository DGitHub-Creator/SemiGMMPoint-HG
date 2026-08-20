#!/usr/bin/env bash
# SemiGMMPoint ODPT-HG 10%: semi-supervised training (100 epochs, full run).
set -euo pipefail
export BUDGET=10
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_budget.sh" "$@"
