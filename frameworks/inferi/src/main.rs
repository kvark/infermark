//! Inferi framework benchmark runner for inferena.
//!
//! Uses inferi's built-in LLaMA transformer with wgpu (Vulkan/Metal) or CUDA GPU backend.
//! Processes tokens sequentially with KV cache, matching inferi's native inference pattern.

use inferi::context::{LlmContext, LlmOps};
use inferi::models::llama2::cpu::Llama2Config;
use inferi::models::llama2::{Llama2, Llama2State, Llama2Weights, LlamaModelType};
use inferi::re_exports::khal::backend::{Backend, GpuBackend};
use inferi::re_exports::vortx::shapes::TensorLayoutBuffers;
use inferi::tensor_cache::TensorCache;
use nalgebra::DVector;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

fn smollm2_config() -> Llama2Config {
    Llama2Config {
        hidden_size: 576,
        intermediate_size: 1536,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        vocab_size: 49152,
        max_position_embeddings: 2048,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
    }
}

fn sha256_f32(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &v in data {
        hasher.update(v.to_le_bytes());
    }
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

fn find_model_path(model_name: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().unwrap_or_default();
    let mut root = exe
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    for _ in 0..5 {
        let p = root
            .join("models")
            .join(model_name)
            .join("model.safetensors");
        if p.exists() {
            return Some(p);
        }
        if !root.pop() {
            break;
        }
    }
    let p = PathBuf::from("models")
        .join(model_name)
        .join("model.safetensors");
    if p.exists() {
        return Some(p);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn emit_result(
    model_name: &str,
    backend_name: &str,
    compile_s: f64,
    forward_ms: f64,
    latency_ms: f64,
    logits_data: &[f32],
    loss: f64,
) {
    let logits_hash = sha256_f32(logits_data);
    let logits_sample: Vec<f64> = logits_data.iter().take(16).map(|&v| v as f64).collect();

    let result = serde_json::json!({
        "framework": "inferi",
        "framework_rev": std::env::var("FRAMEWORK_REV").unwrap_or_default(),
        "model": model_name,
        "device": backend_name,
        "gpu_name": backend_name.to_lowercase(),
        "backend": backend_name,
        "timings": {
            "compile_s": (compile_s * 100.0).round() / 100.0,
            "inference_ms": (forward_ms * 1000.0).round() / 1000.0,
            "latency_ms": (latency_ms * 1000.0).round() / 1000.0,
            "training_ms": 0.0,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": (loss * 1_000_000.0).round() / 1_000_000.0,
        },
    });

    println!("{}", serde_json::to_string(&result).unwrap());
}

async fn init_gpu() -> anyhow::Result<GpuBackend> {
    #[cfg(feature = "cuda")]
    {
        use inferi::re_exports::khal::backend::Cuda;
        match Cuda::new(0) {
            Ok(cuda) => {
                eprintln!("[inferi] using CUDA backend");
                return Ok(GpuBackend::Cuda(cuda));
            }
            Err(e) => {
                eprintln!("[inferi] CUDA init failed ({e}), falling back to WebGPU/Vulkan");
            }
        }
    }

    use inferi::re_exports::khal::backend::WebGpu;
    use inferi::re_exports::khal::re_exports::wgpu;

    let features = wgpu::Features::default();
    let limits = wgpu::Limits {
        max_buffer_size: 2_000_000_000,
        max_storage_buffer_binding_size: 2_000_000_000,
        ..Default::default()
    };
    let mut webgpu = WebGpu::new(features, limits).await?;
    webgpu.force_buffer_copy_src = true;
    eprintln!("[inferi] using WebGPU/Vulkan backend");
    Ok(GpuBackend::WebGpu(webgpu))
}

fn detect_backend_name(backend: &GpuBackend) -> &'static str {
    match backend {
        #[cfg(feature = "cuda")]
        GpuBackend::Cuda(_) => "CUDA",
        _ => {
            if cfg!(target_os = "macos") {
                "Metal"
            } else {
                "Vulkan"
            }
        }
    }
}

/// Run a single-token forward pass through the transformer and read back logits.
async fn forward_one_token(
    backend: &GpuBackend,
    ops: &LlmOps,
    transformer: &Llama2,
    state: &mut Llama2State,
    weights: &Llama2Weights,
    config: &Llama2Config,
    shapes: &mut TensorLayoutBuffers,
    tensor_cache: &mut TensorCache,
    pos: u32,
    token: u32,
    logits_out: Option<&mut DVector<f32>>,
) -> anyhow::Result<()> {
    let (rope_config, rms_norm_config, attn_params) = config.derived_configs(pos);

    // Write configs and token embedding to GPU.
    let mut encoder = backend.begin_encoding();
    backend.write_buffer(state.rope_config_mut().buffer_mut(), 0, &[rope_config])?;
    backend.write_buffer(state.rms_norm_config_mut().buffer_mut(), 0, &[rms_norm_config])?;
    backend.write_buffer(state.attn_params_mut().buffer_mut(), 0, &[attn_params])?;
    state
        .x
        .copy_from_view(&mut encoder, weights.token_embd.row(token))?;
    backend.submit(encoder)?;

    // Run the transformer.
    shapes.clear_tmp();
    let mut ctxt = LlmContext {
        backend,
        shapes,
        cache: tensor_cache,
        pass: None,
        encoder: None,
        ops,
    };
    ctxt.begin_submission();
    transformer.launch(&mut ctxt, state, weights, config, &attn_params, pos)?;
    drop(ctxt.pass.take());

    // Read back logits if requested.
    if let Some(out) = logits_out {
        let (logits, readback) = state.logits_and_readback_mut();
        readback.copy_from_view(ctxt.encoder.as_mut().unwrap(), logits)?;
        ctxt.submit();
        backend
            .read_buffer(state.logits_readback().buffer(), out.as_mut_slice())
            .await?;
    } else {
        ctxt.submit();
    }

    Ok(())
}

async fn bench_smollm2(model_name: &str) -> anyhow::Result<()> {
    let config = smollm2_config();
    let seq_len: usize = 128;

    // --- Compile phase: GPU init + model loading ---
    let compile_start = Instant::now();

    eprintln!("[inferi] initializing GPU...");
    let backend = init_gpu().await?;
    let backend_name = detect_backend_name(&backend);

    eprintln!("[inferi] creating ops and transformer...");
    let ops = LlmOps::new(&backend)?;
    let transformer = Llama2::new(&backend, LlamaModelType::Llama)?;

    // Load weights from safetensors.
    let path = find_model_path(model_name);
    if path.is_none() {
        anyhow::bail!(
            "model weights not found for {model_name}. Run with --download to fetch them."
        );
    }
    let path = path.unwrap();
    eprintln!("[inferi] loading weights from {}", path.display());
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = inferi::re_exports::safetensors::SafeTensors::deserialize(&mmap)?;
    let weights = Llama2Weights::from_safetensors(&backend, &config, &st)?;

    let mut state = Llama2State::new(&backend, &config)?;
    let mut shapes = TensorLayoutBuffers::new(&backend);
    let mut tensor_cache = TensorCache::default();

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[inferi] compiled in {compile_s:.2}s ({backend_name})");

    // --- Inference: process all tokens sequentially with KV cache ---
    let mut logits = DVector::zeros(config.vocab_size);

    eprintln!("[inferi] running inference ({seq_len} tokens)...");
    let fwd_start = Instant::now();
    for pos in 0..seq_len {
        let token = pos as u32;
        let read_logits = pos == seq_len - 1;
        forward_one_token(
            &backend,
            &ops,
            &transformer,
            &mut state,
            &weights,
            &config,
            &mut shapes,
            &mut tensor_cache,
            pos as u32,
            token,
            if read_logits {
                Some(&mut logits)
            } else {
                None
            },
        )
        .await?;
    }
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    let logits_data: Vec<f32> = logits.as_slice().to_vec();
    eprintln!(
        "[inferi] forward: {forward_ms:.2}ms, {} logits",
        logits_data.len()
    );

    // --- Cross-entropy loss on last position ---
    let vocab = config.vocab_size;
    let target = seq_len % vocab;
    let max_logit = logits_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits_data
        .iter()
        .map(|&l| ((l - max_logit) as f64).exp())
        .sum();
    let loss = -((logits_data[target] - max_logit) as f64 - sum_exp.ln());

    // --- Latency: single-token forward ---
    // Reset state for a fresh run.
    let mut lat_state = Llama2State::new(&backend, &config)?;
    let mut lat_shapes = TensorLayoutBuffers::new(&backend);
    let mut lat_cache = TensorCache::default();

    // Warm-up.
    forward_one_token(
        &backend,
        &ops,
        &transformer,
        &mut lat_state,
        &weights,
        &config,
        &mut lat_shapes,
        &mut lat_cache,
        0,
        0,
        None,
    )
    .await?;

    // Fresh state for measurement.
    let mut lat_state = Llama2State::new(&backend, &config)?;
    let mut lat_shapes = TensorLayoutBuffers::new(&backend);
    let mut lat_cache = TensorCache::default();

    let lat_start = Instant::now();
    forward_one_token(
        &backend,
        &ops,
        &transformer,
        &mut lat_state,
        &weights,
        &config,
        &mut lat_shapes,
        &mut lat_cache,
        0,
        0,
        None,
    )
    .await?;
    let latency_ms = lat_start.elapsed().as_secs_f64() * 1000.0;

    emit_result(
        model_name,
        backend_name,
        compile_s,
        forward_ms,
        latency_ms,
        &logits_data,
        loss,
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "SmolLM2-135M".to_string());

    let supported = ["SmolLM2-135M"];
    if !supported.contains(&model_name.as_str()) {
        eprintln!(
            "[inferi] Unknown model: {model_name}. Supported: {}",
            supported.join(", ")
        );
        std::process::exit(1);
    }

    if std::env::var("INFERENA_DRY_RUN").as_deref() == Ok("1") {
        eprintln!("[inferi] dry-run OK: {model_name}");
        return Ok(());
    }

    match model_name.as_str() {
        "SmolLM2-135M" => {
            pollster::block_on(bench_smollm2(&model_name))?;
        }
        _ => unreachable!(),
    }

    Ok(())
}
