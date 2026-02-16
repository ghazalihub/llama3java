use rand::Rng;
use crate::tensors::softmax;

pub struct Sampler {
    rng: rand::rngs::StdRng,
}

impl Sampler {
    pub fn new(seed: Option<u64>) -> Self {
        let rng = if let Some(s) = seed {
            rand::SeedableRng::seed_from_u64(s)
        } else {
            rand::SeedableRng::from_entropy()
        };
        Sampler { rng }
    }

    pub fn sample(&mut self, logits: &mut [f32], temp: f32, topp: f32) -> u32 {
        if temp == 0.0 {
            return logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
        }

        for v in logits.iter_mut() { *v /= temp; }
        softmax(logits);

        if topp <= 0.0 || topp >= 1.0 {
            let r: f32 = self.rng.gen();
            let mut cdf = 0.0;
            for (i, &p) in logits.iter().enumerate() {
                cdf += p;
                if r < cdf { return i as u32; }
            }
            return (logits.len() - 1) as u32;
        }

        self.sample_topp(logits, topp)
    }

    fn sample_topp(&mut self, logits: &[f32], topp: f32) -> u32 {
        let threshold = (1.0 - topp) / (logits.len() as f32);
        let mut candidates: Vec<(usize, f32)> = logits.iter().enumerate()
            .filter(|&(_, &p)| p > threshold)
            .map(|(i, &p)| (i, p))
            .collect();

        if candidates.is_empty() {
            return logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumulative_prob = 0.0;
        let mut last_idx = candidates.len() - 1;
        for (i, &(_, p)) in candidates.iter().enumerate() {
            cumulative_prob += p;
            if cumulative_prob > topp {
                last_idx = i;
                break;
            }
        }

        let r: f32 = self.rng.gen::<f32>() * cumulative_prob;
        let mut cdf = 0.0;
        for &(i, p) in candidates[..=last_idx].iter() {
            cdf += p;
            if r < cdf { return i as u32; }
        }
        candidates[last_idx].0 as u32
    }
}
