#!/usr/bin/env bash
# SemiGMMPoint ODPT 10%: official Area_3 eval on the final checkpoint.
set -euo pipefail
export BUDGET=10
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_budget.sh" "$@"
