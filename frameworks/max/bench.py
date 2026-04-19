#!/usr/bin/env python3
"""MAX (Modular) benchmark runner for inferena.

Builds models using MAX's Graph API + ops (matmul, silu, softmax, rms_norm,
rope, etc.). MAX's MLIR compiler JIT-compiles the graph. Training/backward
is not supported — MAX is an inference-only framework (training_ms = 0).

Explicit f32 throughout — MAX supports bf16/fp16 but we pin everything to
float32 to ensure valid cross-framework numerical comparison.
"""

import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np

from max.dtype import DType
DTYPE = DType.float32
NP_DTYPE = np.float32

SMOLLM2_CONFIG = {
    "hidden_size": 576, "intermediate_size": 1536, "num_hidden_layers": 30,
    "num_attention_heads": 9, "num_key_value_heads": 3, "vocab_size": 49152,
    "max_position_embeddings": 2048, "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
}
SEQ_LEN = 128

SMOLVLA_CONFIG = {
    "action_dim": 32, "expert_hidden": 720, "intermediate": 2048,
    "num_layers": 16, "num_heads": 15, "num_kv_heads": 5, "head_dim": 64,
    "vlm_kv_dim": 320, "self_attn_every_n": 2, "rms_norm_eps": 1e-5,
}
SMOLVLA_CHUNK_SIZE = 50
SMOLVLA_VLM_SEQ_LEN = 16


def sha256_f32(data: np.ndarray) -> str:
    flat = data.astype(np.float32).flatten()
    raw = struct.pack(f"<{flat.size}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def cross_entropy_np(logits_2d, labels_1d):
    total = 0.0
    for i in range(logits_2d.shape[0]):
        row = logits_2d[i].astype(np.float64)
        mx = row.max()
        lse = mx + math.log(np.exp(row - mx).sum())
        total += -(row[labels_1d[i]] - lse)
    return total / logits_2d.shape[0]


def find_model_path(model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    p = os.path.join(root_dir, "models", model_name, "model.safetensors")
    return p if os.path.isfile(p) else None


def emit(model_name, compile_s, inference_ms, latency_ms, output, loss, backend):
    out_np = np.asarray(output, dtype=np.float32)
    logits_hash = sha256_f32(out_np)
    logits_sample = [round(float(v), 6) for v in out_np.flatten()[:16]]
    result = {
        "framework": "max", "model": model_name,
        "device": backend.lower(), "gpu_name": backend.lower(), "backend": backend,
        "timings": {
            "compile_s": round(compile_s, 2),
            "inference_ms": round(inference_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "training_ms": 0.0,  # MAX is inference-only
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": round(float(loss), 6),
        },
    }
    print(json.dumps(result))


def _load_weights(path):
    if path is None:
        return {}
    import safetensors.numpy
    return safetensors.numpy.load_file(path)


def _get(st, name, shape):
    if name in st:
        w = st[name].astype(NP_DTYPE)
        return w.reshape(shape) if w.shape != tuple(shape) else w
    return np.zeros(shape, dtype=NP_DTYPE)


def _name_seed(name: str) -> float:
    h = 0
    for c in name.encode('ascii'):
        h = ((h * 31) + c) & 0xFFFFFFFF
    return float(h % 10000)


def _init_param(name, shape, scale=0.1):
    """Deterministic init matching PyTorch/JAX/meganeura: sin(i*0.01 + name_seed) * scale."""
    seed = _name_seed(name)
    n = 1
    for s in shape:
        n *= s
    return (np.sin(np.arange(n, dtype=NP_DTYPE) * 0.01 + seed) * scale).reshape(shape)


def _init_transposed(name, out_dim, in_dim, scale=0.1):
    """PyTorch Linear weights: stored as [out, in]."""
    seed = _name_seed(name)
    w = np.sin(np.arange(out_dim * in_dim, dtype=NP_DTYPE) * 0.01 + seed) * scale
    return w.reshape(out_dim, in_dim)


def _detect_backend():
    try:
        from max.driver import Accelerator
        Accelerator()
        return "GPU"
    except Exception:
        return "CPU"


# ── SmolLM2-135M ──────────────────────────────────────────────────

def _build_smollm2_graph(st, sl):
    """Build a SmolLM2 forward graph for a given sequence length."""
    from max.graph import Graph, TensorType, ops, DeviceRef

    C = SMOLLM2_CONFIG
    dim, n_layers = C["hidden_size"], C["num_hidden_layers"]
    n_heads, n_kv = C["num_attention_heads"], C["num_key_value_heads"]
    vocab, ffn = C["vocab_size"], C["intermediate_size"]
    hd = dim // n_heads
    kv_dim = n_kv * hd
    eps, theta = C["rms_norm_eps"], C["rope_theta"]
    kv_rep = n_heads // n_kv
    scale_val = 1.0 / math.sqrt(hd)
    dev_cpu = DeviceRef.CPU()

    # Precompute RoPE cos/sin tables and causal mask.
    pos_range = np.arange(sl, dtype=NP_DTYPE)[:, None]
    dim_range = np.arange(hd // 2, dtype=NP_DTYPE)[None, :]
    freqs = pos_range / (theta ** (2.0 * dim_range / hd))
    cos_np = np.cos(freqs).astype(NP_DTYPE)
    sin_np = np.sin(freqs).astype(NP_DTYPE)

    mask_np = np.triu(np.full((sl, sl), -1e9, dtype=NP_DTYPE), k=1).reshape(1, 1, sl, sl)

    # Tied embedding fallback for lm_head.
    lm_w = _get(st, "lm_head.weight", [vocab, dim])
    if np.all(lm_w == 0) and "model.embed_tokens.weight" in st:
        lm_w = st["model.embed_tokens.weight"].astype(NP_DTYPE)

    def const(arr):
        return ops.constant(arr, device=dev_cpu)

    def scalar(v):
        return ops.constant(v, DTYPE, device=dev_cpu)

    def forward(input_ids):
        embed = const(_get(st, "model.embed_tokens.weight", [vocab, dim]))
        h = ops.unsqueeze(ops.gather(embed, input_ids[0], axis=0), 0)
        rc = ops.unsqueeze(ops.unsqueeze(const(cos_np), 0), 0)
        rs = ops.unsqueeze(ops.unsqueeze(const(sin_np), 0), 0)
        cm = const(mask_np)

        for li in range(n_layers):
            pf = f"model.layers.{li}"
            nw = const(_get(st, f"{pf}.input_layernorm.weight", [dim]))
            var = ops.mean(h * h, axis=-1)
            xn = h * ops.rsqrt(var + scalar(eps)) * nw

            wq = const(_get(st, f"{pf}.self_attn.q_proj.weight", [dim, dim]))
            wk = const(_get(st, f"{pf}.self_attn.k_proj.weight", [kv_dim, dim]))
            wv = const(_get(st, f"{pf}.self_attn.v_proj.weight", [kv_dim, dim]))

            q = ops.transpose(ops.reshape(ops.matmul(xn, ops.transpose(wq, -1, -2)), [1, sl, n_heads, hd]), 1, 2)
            k = ops.transpose(ops.reshape(ops.matmul(xn, ops.transpose(wk, -1, -2)), [1, sl, n_kv, hd]), 1, 2)
            v = ops.transpose(ops.reshape(ops.matmul(xn, ops.transpose(wv, -1, -2)), [1, sl, n_kv, hd]), 1, 2)

            q1, q2 = ops.chunk(q, 2, axis=-1)
            k1, k2 = ops.chunk(k, 2, axis=-1)
            q = ops.concat([q1 * rc - q2 * rs, q2 * rc + q1 * rs], axis=-1)
            k = ops.concat([k1 * rc - k2 * rs, k2 * rc + k1 * rs], axis=-1)

            if kv_rep > 1:
                k = ops.repeat_interleave(k, kv_rep, axis=1)
                v = ops.repeat_interleave(v, kv_rep, axis=1)

            scores = ops.matmul(q, ops.transpose(k, -1, -2)) * scalar(scale_val)
            attn = ops.softmax(scores + cm)
            ao = ops.reshape(ops.transpose(ops.matmul(attn, v), 1, 2), [1, sl, dim])

            wo = const(_get(st, f"{pf}.self_attn.o_proj.weight", [dim, dim]))
            h = h + ops.matmul(ao, ops.transpose(wo, -1, -2))

            nw2 = const(_get(st, f"{pf}.post_attention_layernorm.weight", [dim]))
            var2 = ops.mean(h * h, axis=-1)
            xn2 = h * ops.rsqrt(var2 + scalar(eps)) * nw2

            wg = const(_get(st, f"{pf}.mlp.gate_proj.weight", [ffn, dim]))
            wu = const(_get(st, f"{pf}.mlp.up_proj.weight", [ffn, dim]))
            wd = const(_get(st, f"{pf}.mlp.down_proj.weight", [dim, ffn]))
            gate = ops.matmul(xn2, ops.transpose(wg, -1, -2))
            up = ops.matmul(xn2, ops.transpose(wu, -1, -2))
            h = h + ops.matmul(ops.silu(gate) * up, ops.transpose(wd, -1, -2))

        fnw = const(_get(st, "model.norm.weight", [dim]))
        fvar = ops.mean(h * h, axis=-1)
        h = h * ops.rsqrt(fvar + scalar(eps)) * fnw
        return ops.matmul(h, ops.transpose(const(lm_w), -1, -2))

    return Graph(
        f"smollm2_seq{sl}", forward,
        input_types=[TensorType(DType.int64, [1, sl], device=dev_cpu)],
    )


def bench_smollm2(model_name):
    from max.engine import InferenceSession
    from max.driver import CPU

    weights_path = find_model_path(model_name)
    print(f"[max] weights: {weights_path or 'zeros'}", file=sys.stderr)
    st = _load_weights(weights_path)
    if st:
        print(f"[max] loaded {len(st)} tensors", file=sys.stderr)

    session = InferenceSession(devices=[CPU()])

    print(f"[max] building + compiling seq_len={SEQ_LEN} graph...", file=sys.stderr)
    t0 = time.perf_counter()
    model = session.load(_build_smollm2_graph(st, SEQ_LEN))
    compile_s = time.perf_counter() - t0
    print(f"[max] compiled in {compile_s:.2f}s", file=sys.stderr)

    vocab = SMOLLM2_CONFIG["vocab_size"]
    input_ids = np.arange(SEQ_LEN, dtype=np.int64).reshape(1, SEQ_LEN)
    labels = np.array([(i + 1) % vocab for i in range(SEQ_LEN)], dtype=np.int64)

    model(input_ids)  # warm-up
    t0 = time.perf_counter()
    result = model(input_ids)
    inference_ms = (time.perf_counter() - t0) * 1000.0
    logits = result[0].to_numpy()
    print(f"[max] inference: {inference_ms:.2f}ms, logits: {logits.shape}", file=sys.stderr)

    loss = cross_entropy_np(logits[0], labels)

    # Single-token latency: compile a separate seq_len=1 graph.  The extra
    # compile time is NOT counted in compile_s (matches JAX's approach).
    print("[max] building seq_len=1 graph for latency...", file=sys.stderr)
    model_lat = session.load(_build_smollm2_graph(st, 1))
    lat_input = np.array([[0]], dtype=np.int64)
    model_lat(lat_input)  # warm-up
    t0 = time.perf_counter()
    model_lat(lat_input)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    emit(model_name, compile_s, inference_ms, latency_ms, logits, loss, _detect_backend())


# ── SmolVLA Action Expert ────────────────────────────────────────

def _build_smolvla_graph():
    from max.graph import Graph, TensorType, ops, DeviceRef

    C = SMOLVLA_CONFIG
    dim = C["expert_hidden"]
    inter = C["intermediate"]
    n_layers = C["num_layers"]
    n_heads, n_kv = C["num_heads"], C["num_kv_heads"]
    hd = C["head_dim"]
    kv_rep = n_heads // n_kv
    action_dim = C["action_dim"]
    vlm_kv_dim = C["vlm_kv_dim"]
    self_attn_every_n = C["self_attn_every_n"]
    eps = C["rms_norm_eps"]
    scale_val = 1.0 / math.sqrt(hd)
    cs = SMOLVLA_CHUNK_SIZE
    vs = SMOLVLA_VLM_SEQ_LEN
    dev_cpu = DeviceRef.CPU()

    def const(arr):
        return ops.constant(arr, device=dev_cpu)

    def scalar(v):
        return ops.constant(v, DTYPE, device=dev_cpu)

    # Deterministic random init (SmolVLA has no pretrained weights to load).
    w_action_proj = _init_transposed("action_proj.weight", dim, action_dim)
    w_time_proj = _init_transposed("time_proj.weight", dim, dim * 2)
    w_kv_proj = _init_transposed("kv_proj.weight", dim, vlm_kv_dim)
    w_final_norm = _init_param("norm.weight", (dim,))
    w_head = _init_transposed("head.weight", action_dim, dim)

    layer_weights = [{
        "norm1": _init_param(f"layers.{i}.norm1.weight", (dim,)),
        "wq": _init_transposed(f"layers.{i}.attn.q_proj.weight", n_heads * hd, dim),
        "wk": _init_transposed(f"layers.{i}.attn.k_proj.weight", n_kv * hd, dim),
        "wv": _init_transposed(f"layers.{i}.attn.v_proj.weight", n_kv * hd, dim),
        "wo": _init_transposed(f"layers.{i}.attn.o_proj.weight", dim, n_heads * hd),
        "norm2": _init_param(f"layers.{i}.norm2.weight", (dim,)),
        "w_gate": _init_transposed(f"layers.{i}.mlp.gate.weight", inter, dim),
        "w_up": _init_transposed(f"layers.{i}.mlp.up.weight", inter, dim),
        "w_down": _init_transposed(f"layers.{i}.mlp.down.weight", dim, inter),
    } for i in range(n_layers)]

    def rms_norm(x, w):
        var = ops.mean(x * x, axis=-1)
        return x * ops.rsqrt(var + scalar(eps)) * w

    def gq_attention(q_input, kv_input, wq, wk, wv, wo, sq, sk):
        q = ops.matmul(q_input, ops.transpose(wq, -1, -2))
        k = ops.matmul(kv_input, ops.transpose(wk, -1, -2))
        v = ops.matmul(kv_input, ops.transpose(wv, -1, -2))
        q = ops.transpose(ops.reshape(q, [1, sq, n_heads, hd]), 1, 2)
        k = ops.transpose(ops.reshape(k, [1, sk, n_kv, hd]), 1, 2)
        v = ops.transpose(ops.reshape(v, [1, sk, n_kv, hd]), 1, 2)
        if kv_rep > 1:
            k = ops.repeat_interleave(k, kv_rep, axis=1)
            v = ops.repeat_interleave(v, kv_rep, axis=1)
        # No causal mask — action expert is non-autoregressive.
        scores = ops.matmul(q, ops.transpose(k, -1, -2)) * scalar(scale_val)
        attn = ops.softmax(scores)
        out = ops.reshape(ops.transpose(ops.matmul(attn, v), 1, 2), [1, sq, n_heads * hd])
        return ops.matmul(out, ops.transpose(wo, -1, -2))

    def forward(noisy_actions, timestep, vlm_kv):
        # noisy_actions: [1, cs, action_dim]
        # timestep:      [1, 1, dim*2]
        # vlm_kv:        [1, vs, vlm_kv_dim]
        x = ops.matmul(noisy_actions, ops.transpose(const(w_action_proj), -1, -2))
        t = ops.matmul(timestep, ops.transpose(const(w_time_proj), -1, -2))
        x = x + t  # broadcast [1,1,dim] over [1,cs,dim]
        kv = ops.matmul(vlm_kv, ops.transpose(const(w_kv_proj), -1, -2))

        for i in range(n_layers):
            lw = layer_weights[i]
            h = rms_norm(x, const(lw["norm1"]))
            if i % self_attn_every_n == 0:
                attn_out = gq_attention(h, h, const(lw["wq"]), const(lw["wk"]),
                                        const(lw["wv"]), const(lw["wo"]), cs, cs)
            else:
                attn_out = gq_attention(h, kv, const(lw["wq"]), const(lw["wk"]),
                                        const(lw["wv"]), const(lw["wo"]), cs, vs)
            x = x + attn_out

            h2 = rms_norm(x, const(lw["norm2"]))
            gate = ops.matmul(h2, ops.transpose(const(lw["w_gate"]), -1, -2))
            up = ops.matmul(h2, ops.transpose(const(lw["w_up"]), -1, -2))
            x = x + ops.matmul(ops.silu(gate) * up, ops.transpose(const(lw["w_down"]), -1, -2))

        x = rms_norm(x, const(w_final_norm))
        return ops.matmul(x, ops.transpose(const(w_head), -1, -2))

    return Graph(
        "smolvla", forward,
        input_types=[
            TensorType(DTYPE, [1, cs, action_dim], device=dev_cpu),
            TensorType(DTYPE, [1, 1, dim * 2], device=dev_cpu),
            TensorType(DTYPE, [1, vs, vlm_kv_dim], device=dev_cpu),
        ],
    )


def bench_smolvla(model_name):
    from max.engine import InferenceSession
    from max.driver import CPU

    print("[max] building SmolVLA action expert (random-init)...", file=sys.stderr)
    t0 = time.perf_counter()
    session = InferenceSession(devices=[CPU()])
    model = session.load(_build_smolvla_graph())
    compile_s = time.perf_counter() - t0
    print(f"[max] compiled in {compile_s:.2f}s", file=sys.stderr)

    C = SMOLVLA_CONFIG
    dim = C["expert_hidden"]
    action_dim = C["action_dim"]
    vlm_kv_dim = C["vlm_kv_dim"]
    cs, vs = SMOLVLA_CHUNK_SIZE, SMOLVLA_VLM_SEQ_LEN

    # Deterministic inputs matching PyTorch/JAX.
    noisy = np.sin(np.arange(cs * action_dim, dtype=NP_DTYPE) * 0.01).reshape(1, cs, action_dim)
    timestep = np.sin(np.arange(dim * 2, dtype=NP_DTYPE) * 0.005).reshape(1, 1, dim * 2)
    vlm_kv = np.cos(np.arange(vs * vlm_kv_dim, dtype=NP_DTYPE) * 0.01).reshape(1, vs, vlm_kv_dim)

    model(noisy, timestep, vlm_kv)  # warm-up
    t0 = time.perf_counter()
    result = model(noisy, timestep, vlm_kv)
    inference_ms = (time.perf_counter() - t0) * 1000.0
    pred = result[0].to_numpy()
    print(f"[max] inference: {inference_ms:.2f}ms, output: {pred.shape}", file=sys.stderr)

    loss = float(np.mean(pred.astype(np.float64) ** 2))

    # No single-item latency for SmolVLA: action expert processes the full
    # chunk at once (non-autoregressive), so per-token latency isn't meaningful.
    latency_ms = 0.0

    emit(model_name, compile_s, inference_ms, latency_ms, pred, loss, _detect_backend())


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    supported = {
        "SmolLM2-135M": bench_smollm2,
        "SmolVLA": bench_smolvla,
    }
    if model_name not in supported:
        print(f"[max] Unknown model: {model_name}. Supported: {', '.join(supported.keys())}",
              file=sys.stderr)
        sys.exit(1)
    supported[model_name](model_name)
