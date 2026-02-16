use crate::gguf::{GGUF, MetadataValue};

#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub head_size: usize,
    pub rope_scaling: bool,
    pub scale_factor: f32,
    pub lo_freq_factor: f32,
    pub hi_freq_factor: f32,
    pub old_context_len: f32,
}

impl Config {
    pub fn from_gguf(gguf: &GGUF) -> Self {
        let dim = get_uint32(&gguf.metadata, "llama.embedding_length") as usize;
        let hidden_dim = get_uint32(&gguf.metadata, "llama.feed_forward_length") as usize;
        let n_layers = get_uint32(&gguf.metadata, "llama.block_count") as usize;
        let n_heads = get_uint32(&gguf.metadata, "llama.attention.head_count") as usize;
        let n_kv_heads = gguf.metadata.get("llama.attention.head_count_kv")
            .map(|v| if let MetadataValue::Uint32(a) = v { *a } else { n_heads as u32 })
            .unwrap_or(n_heads as u32) as usize;

        let vocab_size = if let Some(MetadataValue::Array(tokens)) = gguf.metadata.get("tokenizer.ggml.tokens") {
            tokens.len()
        } else {
            0
        };

        let seq_len = get_uint32(&gguf.metadata, "llama.context_length") as usize;
        let rope_theta = get_float32(&gguf.metadata, "llama.rope.freq_base", 10000.0);
        let eps = get_float32(&gguf.metadata, "llama.attention.layer_norm_rms_epsilon", 1e-5);

        let mut rope_scaling = false;
        let mut scale_factor = 8.0;
        let mut lo_freq_factor = 1.0;
        let mut hi_freq_factor = 3.0;
        let mut old_context_len = 8192.0;

        if let Some(MetadataValue::String(rst)) = gguf.metadata.get("llama.rope.scaling.type") {
            if rst == "linear" || rst == "yarn" {
                rope_scaling = true;
            }
        }
        scale_factor = get_float32(&gguf.metadata, "llama.rope.scaling.factor", 8.0);
        lo_freq_factor = get_float32(&gguf.metadata, "llama.rope.scaling.low_freq_factor", 1.0);
        hi_freq_factor = get_float32(&gguf.metadata, "llama.rope.scaling.high_freq_factor", 3.0);
        old_context_len = get_uint32_opt(&gguf.metadata, "llama.rope.scaling.orig_ctx_len").map(|v| v as f32).unwrap_or(8192.0);

        Config {
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len,
            rope_theta, eps, head_size: dim / n_heads,
            rope_scaling, scale_factor, lo_freq_factor, hi_freq_factor, old_context_len,
        }
    }
}

fn get_uint32(metadata: &std::collections::HashMap<String, MetadataValue>, key: &str) -> u32 {
    if let Some(MetadataValue::Uint32(v)) = metadata.get(key) {
        *v
    } else {
        panic!("Missing or invalid key: {}", key);
    }
}

fn get_uint32_opt(metadata: &std::collections::HashMap<String, MetadataValue>, key: &str) -> Option<u32> {
    if let Some(MetadataValue::Uint32(v)) = metadata.get(key) {
        Some(*v)
    } else {
        None
    }
}

fn get_float32(metadata: &std::collections::HashMap<String, MetadataValue>, key: &str, default: f32) -> f32 {
    if let Some(MetadataValue::Float32(v)) = metadata.get(key) {
        *v
    } else {
        default
    }
}
