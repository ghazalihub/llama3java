use crate::gguf::{GGUF, GGMLType};
use crate::config::Config;
use crate::tensors::{Tensor, rmsnorm, matmul, dot_f32};
use crate::tokenizer::Tokenizer;
use crate::sampler::Sampler;
use std::time::Instant;

pub struct Weights<'a> {
    pub token_embd: Tensor<'a>,
    pub rms_att_w: Vec<Tensor<'a>>,
    pub wq: Vec<Tensor<'a>>,
    pub wk: Vec<Tensor<'a>>,
    pub wv: Vec<Tensor<'a>>,
    pub wo: Vec<Tensor<'a>>,
    pub rms_ffn_w: Vec<Tensor<'a>>,
    pub w1: Vec<Tensor<'a>>,
    pub w2: Vec<Tensor<'a>>,
    pub w3: Vec<Tensor<'a>>,
    pub rms_final_w: Tensor<'a>,
    pub output_w: Tensor<'a>,
}

impl<'a> Weights<'a> {
    pub fn load(config: &Config, gguf: &'a GGUF, mmap_data: &'a [u8]) -> Self {
        let mut rms_att_w = Vec::with_capacity(config.n_layers);
        let mut wq = Vec::with_capacity(config.n_layers);
        let mut wk = Vec::with_capacity(config.n_layers);
        let mut wv = Vec::with_capacity(config.n_layers);
        let mut wo = Vec::with_capacity(config.n_layers);
        let mut rms_ffn_w = Vec::with_capacity(config.n_layers);
        let mut w1 = Vec::with_capacity(config.n_layers);
        let mut w2 = Vec::with_capacity(config.n_layers);
        let mut w3 = Vec::with_capacity(config.n_layers);

        for i in 0..config.n_layers {
            rms_att_w.push(get_tensor(gguf, mmap_data, &format!("blk.{}.attn_norm.weight", i)));
            wq.push(get_tensor(gguf, mmap_data, &format!("blk.{}.attn_q.weight", i)));
            wk.push(get_tensor(gguf, mmap_data, &format!("blk.{}.attn_k.weight", i)));
            wv.push(get_tensor(gguf, mmap_data, &format!("blk.{}.attn_v.weight", i)));
            wo.push(get_tensor(gguf, mmap_data, &format!("blk.{}.attn_output.weight", i)));
            rms_ffn_w.push(get_tensor(gguf, mmap_data, &format!("blk.{}.ffn_norm.weight", i)));
            w1.push(get_tensor(gguf, mmap_data, &format!("blk.{}.ffn_gate.weight", i)));
            w2.push(get_tensor(gguf, mmap_data, &format!("blk.{}.ffn_down.weight", i)));
            w3.push(get_tensor(gguf, mmap_data, &format!("blk.{}.ffn_up.weight", i)));
        }

        let token_embd = get_tensor(gguf, mmap_data, "token_embd.weight");
        let rms_final_w = get_tensor(gguf, mmap_data, "output_norm.weight");
        let output_w = if gguf.tensor_infos.contains_key("output.weight") {
            get_tensor(gguf, mmap_data, "output.weight")
        } else {
            // Tie embeddings
            Tensor { data: token_embd.data, ggml_type: token_embd.ggml_type, num_elements: token_embd.num_elements }
        };

        Weights {
            token_embd, rms_att_w, wq, wk, wv, wo, rms_ffn_w, w1, w2, w3, rms_final_w, output_w
        }
    }
}

fn get_tensor<'a>(gguf: &'a GGUF, mmap_data: &'a [u8], name: &str) -> Tensor<'a> {
    let info = gguf.tensor_infos.get(name).expect(&format!("Tensor not found: {}", name));
    let offset = gguf.tensor_data_offset + info.offset;
    let size = info.ggml_type.byte_size_for(info.num_elements());
    Tensor {
        data: &mmap_data[offset as usize..(offset as usize + size)],
        ggml_type: info.ggml_type,
        num_elements: info.num_elements(),
    }
}

pub struct State {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

impl State {
    pub fn new(config: &Config) -> Self {
        let kv_dim = config.n_kv_heads * config.head_size;
        State {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            k: vec![0.0; config.dim],
            v: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
            value_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
        }
    }
}

pub fn forward(config: &Config, state: &mut State, weights: &Weights, token: u32, pos: usize) {
    let dim = config.dim;
    let hdim = config.hidden_dim;
    let hsz = config.head_size;
    let nhead = config.n_heads;
    let nkv = config.n_kv_heads;
    let kv_dim = nkv * hsz;
    let kv_mul = nhead / nkv;

    weights.token_embd.copy_to_f32(token as usize * dim, dim, &mut state.x);

    let mut weight_f32 = vec![0.0; dim.max(hdim)];

    for l in 0..config.n_layers {
        weights.rms_att_w[l].copy_to_f32(0, dim, &mut weight_f32[..dim]);
        rmsnorm(&mut state.xb, &state.x, &weight_f32[..dim], config.eps);

        matmul(&mut state.q, &state.xb, &weights.wq[l], nhead * hsz, dim);
        matmul(&mut state.k, &state.xb, &weights.wk[l], nkv * hsz, dim);
        matmul(&mut state.v, &state.xb, &weights.wv[l], nkv * hsz, dim);

        // RoPE
        for h in 0..nhead {
            for i in 0..hsz / 2 {
                let mut freq = 1.0 / (config.rope_theta.powf((2 * i) as f32 / hsz as f32));
                if config.rope_scaling {
                    let wlen = 2.0 * std::f32::consts::PI / freq;
                    let low = config.old_context_len / config.lo_freq_factor;
                    let high = config.old_context_len / config.hi_freq_factor;
                    if wlen > low { freq /= config.scale_factor; }
                    else if wlen > high {
                        let smooth = (config.old_context_len / wlen - config.lo_freq_factor) / (config.hi_freq_factor - config.lo_freq_factor);
                        freq = (1.0 - smooth) * freq / config.scale_factor + smooth * freq;
                    }
                }
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();

                let q0 = state.q[h * hsz + i * 2];
                let q1 = state.q[h * hsz + i * 2 + 1];
                state.q[h * hsz + i * 2] = q0 * fcr - q1 * fci;
                state.q[h * hsz + i * 2 + 1] = q0 * fci + q1 * fcr;

                if h < nkv {
                    let k0 = state.k[h * hsz + i * 2];
                    let k1 = state.k[h * hsz + i * 2 + 1];
                    state.k[h * hsz + i * 2] = k0 * fcr - k1 * fci;
                    state.k[h * hsz + i * 2 + 1] = k0 * fci + k1 * fcr;
                }
            }
        }

        let loff = l * config.seq_len * kv_dim;
        state.key_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim].copy_from_slice(&state.k[0..kv_dim]);
        state.value_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim].copy_from_slice(&state.v[0..kv_dim]);

        // Multi-head Attention
        for h in 0..nhead {
            let q_offset = h * hsz;
            let att_offset = h * config.seq_len;
            for t in 0..=pos {
                let k_offset = loff + t * kv_dim + (h / kv_mul) * hsz;
                let score = dot_f32(&state.q[q_offset..q_offset + hsz], &state.key_cache[k_offset..k_offset + hsz]);
                state.att[att_offset + t] = score / (hsz as f32).sqrt();
            }

            crate::tensors::softmax(&mut state.att[att_offset..att_offset + pos + 1]);

            let xb_offset = h * hsz;
            for i in 0..hsz { state.xb[xb_offset + i] = 0.0; }
            for t in 0..=pos {
                let v_offset = loff + t * kv_dim + (h / kv_mul) * hsz;
                let a = state.att[att_offset + t];
                for i in 0..hsz {
                    state.xb[xb_offset + i] += a * state.value_cache[v_offset + i];
                }
            }
        }

        matmul(&mut state.xb2, &state.xb, &weights.wo[l], dim, dim);
        for i in 0..dim { state.x[i] += state.xb2[i]; }

        weights.rms_ffn_w[l].copy_to_f32(0, dim, &mut weight_f32[..dim]);
        rmsnorm(&mut state.xb, &state.x, &weight_f32[..dim], config.eps);

        matmul(&mut state.hb, &state.xb, &weights.w1[l], hdim, dim);
        matmul(&mut state.hb2, &state.xb, &weights.w3[l], hdim, dim);

        for i in 0..hdim {
            let val = state.hb[i];
            state.hb[i] = (val / (1.0 + (-val).exp())) * state.hb2[i];
        }

        matmul(&mut state.xb, &state.hb, &weights.w2[l], dim, hdim);
        for i in 0..dim { state.x[i] += state.xb[i]; }
    }

    weights.rms_final_w.copy_to_f32(0, dim, &mut weight_f32[..dim]);
    let x_clone = state.x.clone();
    rmsnorm(&mut state.x, &x_clone, &weight_f32[..dim], config.eps);

    matmul(&mut state.logits, &state.x, &weights.output_w, config.vocab_size, dim);
}
