use std::collections::HashMap;

pub struct Vocabulary {
    pub tokens: Vec<String>,
    pub token_to_index: HashMap<String, u32>,
}

impl Vocabulary {
    pub fn new(tokens: Vec<String>) -> Self {
        let mut token_to_index = HashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            token_to_index.insert(token.clone(), i as u32);
        }
        Vocabulary { tokens, token_to_index }
    }

    pub fn get_index(&self, token: &str) -> Option<u32> {
        self.token_to_index.get(token).copied()
    }
}

pub struct Tokenizer {
    pub vocab: Vocabulary,
    pub merges: HashMap<(u32, u32), u32>,
    pub special_tokens: HashMap<String, u32>,
    pub b2u: [u32; 256],
    pub u2b: HashMap<u32, u8>,
}

impl Tokenizer {
    pub fn new(vocab: Vocabulary, merges_raw: Vec<String>, special_tokens: HashMap<String, u32>) -> Self {
        let mut merges = HashMap::with_capacity(merges_raw.len());
        for line in merges_raw {
            let parts: Vec<&str> = line.split(' ').collect();
            if parts.len() != 2 { continue; }
            let id1 = vocab.get_index(parts[0]);
            let id2 = vocab.get_index(parts[1]);
            if let (Some(i1), Some(i2)) = (id1, id2) {
                let combined = format!("{}{}", parts[0], parts[1]);
                if let Some(cid) = vocab.get_index(&combined) {
                    merges.insert((i1, i2), cid);
                }
            }
        }

        let mut b2u = [0u32; 256];
        let mut u2b = HashMap::new();

        let mut bs = Vec::new();
        for i in b'!'..=b'~' { bs.push(i); }
        for i in 0xA1..=0xAC { bs.push(i as u8); }
        for i in 0xAE..=0xFF { bs.push(i as u8); }

        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n = 0;
        for b in 0..=255 {
            if !bs.contains(&(b as u8)) {
                bs.push(b as u8);
                cs.push(256 + n);
                n += 1;
            }
        }

        for i in 0..256 {
            b2u[bs[i] as usize] = cs[i];
            u2b.insert(cs[i], bs[i]);
        }

        Tokenizer { vocab, merges, special_tokens, b2u, u2b }
    }

    pub fn encode(&self, text: &str, allowed_special: &[&str]) -> Vec<u32> {
        for &s in allowed_special {
            if text == s {
                if let Some(&id) = self.special_tokens.get(s) {
                    return vec![id];
                }
            }
        }

        let mut ids = Vec::new();
        for &b in text.as_bytes() {
            let token = std::char::from_u32(self.b2u[b as usize]).unwrap().to_string();
            if let Some(id) = self.vocab.get_index(&token) {
                ids.push(id);
            }
        }

        while ids.len() >= 2 {
            let mut best_pair = None;
            let mut best_idx = u32::MAX;

            for i in 0..ids.len() - 1 {
                if let Some(&merge_idx) = self.merges.get(&(ids[i], ids[i+1])) {
                    if merge_idx < best_idx {
                        best_idx = merge_idx;
                        best_pair = Some(i);
                    }
                }
            }

            if let Some(i) = best_pair {
                ids[i] = best_idx;
                ids.remove(i + 1);
            } else {
                break;
            }
        }

        ids
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut res = Vec::new();
        for &t in tokens {
            let s = &self.vocab.tokens[t as usize];
            for c in s.chars() {
                let cp = c as u32;
                if let Some(&b) = self.u2b.get(&cp) {
                    res.push(b);
                } else {
                    res.extend_from_slice(c.to_string().as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&res).to_string()
    }
}
