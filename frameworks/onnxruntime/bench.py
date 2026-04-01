#!/usr/bin/env python3
"""ONNX Runtime benchmark runner for inferena.

Exports the model to ONNX via optimum, then runs inference with
onnxruntime. Training (backward) is not supported.
"""

import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np


MODEL_REGISTRY = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "type": "causal_lm",
    },
}


def sha256_f32(data: np.ndarray) -> str:
    flat = data.astype(np.float32).flatten()
    raw = struct.pack(f"<{flat.size}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(root_dir, "models", model_name)
    onnx_dir = os.path.join(model_dir, "onnx")

    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForCausalLM
    import torch

    seq_len = 128
    vocab_size = 49152

    # --- Export or load ---
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    if not os.path.isfile(onnx_path):
        print(f"[onnxruntime] exporting to ONNX via optimum...", file=sys.stderr)
        os.makedirs(onnx_dir, exist_ok=True)
        ort_model = ORTModelForCausalLM.from_pretrained(model_dir, export=True)
        ort_model.save_pretrained(onnx_dir)
    else:
        print(f"[onnxruntime] loading from {onnx_dir}...", file=sys.stderr)

    t0 = time.perf_counter()
    ort_model = ORTModelForCausalLM.from_pretrained(onnx_dir)
    compile_s = time.perf_counter() - t0
    print(f"[onnxruntime] loaded in {compile_s:.2f}s", file=sys.stderr)

    # --- Prepare inputs ---
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    labels = np.array([[(i + 1) % vocab_size for i in range(seq_len)]], dtype=np.int64)

    # --- Warm-up ---
    ort_model(input_ids=input_ids, attention_mask=attention_mask)

    # --- Inference ---
    t0 = time.perf_counter()
    outputs = ort_model(input_ids=input_ids, attention_mask=attention_mask)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs.logits.detach().numpy()  # (1, seq_len, vocab_size)
    logits_3d = logits[0]  # (seq_len, vocab_size)

    # --- Loss (cross-entropy) ---
    total_loss = 0.0
    for i in range(seq_len):
        row = logits_3d[i].astype(np.float64)
        max_val = row.max()
        log_sum_exp = max_val + math.log(np.exp(row - max_val).sum())
        total_loss += -(row[labels[0, i]] - log_sum_exp)
    loss = total_loss / seq_len

    # --- Latency (single-token forward) ---
    lat_input = torch.zeros(1, 1, dtype=torch.long)
    lat_mask = torch.ones(1, 1, dtype=torch.long)
    ort_model(input_ids=lat_input, attention_mask=lat_mask)
    t0 = time.perf_counter()
    ort_model(input_ids=lat_input, attention_mask=lat_mask)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # --- Logits hash & sample ---
    logits_hash = sha256_f32(logits)
    logits_flat = logits.astype(np.float32).flatten()
    logits_sample = [round(float(v), 6) for v in logits_flat[:16]]

    # --- Output ---
    providers = ort_model.providers if hasattr(ort_model, 'providers') else ["CPU"]
    backend = providers[0] if providers else "CPU"

    result = {
        "framework": "onnxruntime",
        "model": model_name,
        "device": "cpu",
        "gpu_name": "cpu",
        "onnxruntime_version": ort.__version__,
        "backend": backend,
        "timings": {
            "compile_s": round(compile_s, 2),
            "inference_ms": round(inference_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "training_ms": 0.0,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": round(float(loss), 6),
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
