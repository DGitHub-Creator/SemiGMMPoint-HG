#!/usr/bin/env bash
# SemiGMMPoint ODPT 20%: labeling-only training (100 epochs official).
set -euo pipefail
export BUDGET=20
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_budget.sh" "$@"
