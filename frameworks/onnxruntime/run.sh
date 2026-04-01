#!/usr/bin/env bash
# ONNX Runtime benchmark runner for inferena.
# Exports model to ONNX via PyTorch, then benchmarks with onnxruntime.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

# --- Check dependencies ---
python3 -c "import onnxruntime" 2>/dev/null || {
    echo "[onnxruntime] installing onnxruntime..." >&2
    pip install onnxruntime onnx --quiet 2>&1 >&2
}

exec python3 "$SCRIPT_DIR/bench.py" "$MODEL"
