#!/usr/bin/env bash
# SemiGMMPoint ODPT-HG 20%: semi-supervised training (100 epochs official).
set -euo pipefail
export BUDGET=20
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_budget.sh" "$@"
