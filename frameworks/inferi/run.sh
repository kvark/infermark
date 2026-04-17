#!/usr/bin/env bash
# Inferi framework benchmark runner wrapper.
# Uses wgpu (Vulkan/Metal) by default, CUDA with --features cuda.
# Requires `cargo gpu` for shader compilation (install via: cargo install cargo-gpu).
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

source "$ROOT_DIR/scripts/cargo-rev.sh"
export FRAMEWORK_REV=$(cargo_rev_short inferi "$ROOT_DIR")

# Check for cargo-gpu (needed to compile inferi's SPIR-V shaders).
if ! cargo gpu --version &>/dev/null; then
    echo "[inferi] cargo-gpu not found. Install via: cargo install cargo-gpu" >&2
    echo "  (required for compiling inferi's GPU shaders)" >&2
    exit 1
fi

# Select GPU backend based on platform.
FEATURES=""
EXE=""
case "$(uname -s)" in
    Linux*|MINGW*|MSYS*|CYGWIN*)
        case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=.exe ;; esac
        if command -v nvidia-smi &>/dev/null; then
            FEATURES="--features cuda"
        elif command -v rocm-smi &>/dev/null || [ -d /opt/rocm ]; then
            echo "[inferi] AMD GPU detected — using Vulkan via wgpu" >&2
        fi
        ;;
    Darwin*)
        echo "[inferi] macOS detected — using Metal via wgpu" >&2
        ;;
esac

echo "[inferi] Building release binary... $FEATURES" >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p inferena-inferi $FEATURES 2>&1 >&2

exec "$ROOT_DIR/target/release/inferena-inferi${EXE}" "$MODEL"
