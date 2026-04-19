#!/usr/bin/env bash
# MAX (Modular) benchmark runner for inferena.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-SmolLM2-135M}"

: "${PYTHON:?must be set by run.sh}"

# --- Check dependencies ---
if ! "$PYTHON" -c "from max import engine" 2>/dev/null; then
    echo "[max] MAX not found. Install via:" >&2
    echo "  pip install max                        # Linux or macOS (Apple Silicon)" >&2
    echo "  See https://docs.modular.com/max/install" >&2
    exit 1
fi

# Log backend info.
"$PYTHON" -c "
from max.driver import CPU
import sys
try:
    from max.driver import Accelerator
    acc = Accelerator()
    print(f'[max] accelerator available', file=sys.stderr)
except Exception:
    print(f'[max] using CPU backend', file=sys.stderr)
" 2>/dev/null || true

# --- Dry-run: stop after validation ---
if [ "${INFERENA_DRY_RUN:-}" = "1" ]; then
    echo "[max] dry-run OK: $MODEL" >&2
    exit 0
fi

exec "$PYTHON" "$SCRIPT_DIR/bench.py" "$MODEL"
