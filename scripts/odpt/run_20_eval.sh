#!/usr/bin/env bash
# SemiGMMPoint ODPT 20%: official Area_3 eval on the final checkpoint.
set -euo pipefail
export BUDGET=20
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_budget.sh" "$@"
