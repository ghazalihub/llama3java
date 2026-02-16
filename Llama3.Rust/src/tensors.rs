use crate::gguf::GGMLType;
use half::f16;
use rayon::prelude::*;

pub struct Tensor<'a> {
    pub data: &'a [u8],
    pub ggml_type: GGMLType,
    pub num_elements: usize,
}

impl<'a> Tensor<'a> {
    pub fn copy_to_f32(&self, offset: usize, count: usize, out: &mut [f32]) {
        match self.ggml_type {
            GGMLType::F32 => {
                let f_data: &[f32] = unsafe {
                    std::slice::from_raw_parts(self.data.as_ptr().add(offset * 4) as *const f32, count)
                };
                out.copy_from_slice(f_data);
            }
            GGMLType::Q4_0 => {
                for i in (0..count).step_by(32) {
                    let block = &self.data[(offset + i) / 32 * 18..];
                    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..16 {
                        out[i + j] = scale * (((block[2 + j] & 0x0F) as i8 - 8) as f32);
                        out[i + j + 16] = scale * (((block[2 + j] >> 4) as i8 - 8) as f32);
                    }
                }
            }
            GGMLType::Q8_0 => {
                for i in (0..count).step_by(32) {
                    let block = &self.data[(offset + i) / 32 * 34..];
                    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..32 {
                        out[i + j] = scale * (block[2 + j] as i8 as f32);
                    }
                }
            }
            _ => panic!("Unsupported type for copy_to_f32"),
        }
    }
}

pub fn dot(a: &Tensor, a_off: usize, b: &[f32], b_off: usize, n: usize) -> f32 {
    match a.ggml_type {
        GGMLType::F32 => {
            let af: &[f32] = unsafe {
                std::slice::from_raw_parts(a.data.as_ptr().add(a_off * 4) as *const f32, n)
            };
            dot_f32(af, &b[b_off..b_off + n])
        }
        GGMLType::Q4_0 => {
            dot_q4_0_f32(&a.data[a_off / 32 * 18..], &b[b_off..b_off + n])
        }
        GGMLType::Q8_0 => {
            dot_q8_0_f32(&a.data[a_off / 32 * 34..], &b[b_off..b_off + n])
        }
        _ => panic!("Unsupported dot product types"),
    }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_f32_avx(a, b) };
        }
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
unsafe fn dot_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum_v = _mm256_setzero_ps();
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum_v = _mm256_add_ps(sum_v, _mm256_mul_ps(va, vb));
        i += 8;
    }
    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum_v);
    let mut sum: f32 = temp.iter().sum();
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

fn dot_q4_0_f32(a: &[u8], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let n = b.len();
    for i in (0..n).step_by(32) {
        let block = &a[i / 32 * 18..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..16 {
            sum += scale * (((block[2 + j] & 0x0F) as i8 - 8) as f32) * b[i + j];
            sum += scale * (((block[2 + j] >> 4) as i8 - 8) as f32) * b[i + j + 16];
        }
    }
    sum
}

fn dot_q8_0_f32(a: &[u8], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let n = b.len();
    for i in (0..n).step_by(32) {
        let block = &a[i / 32 * 34..];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..32 {
            sum += scale * (block[2 + j] as i8 as f32) * b[i + j];
        }
    }
    sum
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32) + eps;
    let inv_std = 1.0 / ss.sqrt();
    for i in 0..x.len() {
        out[i] = weight[i] * (x[i] * inv_std);
    }
}

pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn matmul(out: &mut [f32], x: &[f32], w: &Tensor, n: usize, d: usize) {
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = dot(w, i * d, x, 0, d);
    });
}
