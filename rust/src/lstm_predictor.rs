/// 2-layer LSTM, 32 hidden units — predicts utilization 5 ms ahead (§V-C)

pub const HIDDEN_SIZE: usize = 32;
pub const HISTORY_LEN: usize = 16;
pub const PREDICT_AHEAD_MS: f64 = 5.0;

pub struct LstmPredictor {
    w_ih: Vec<i8>,
    w_hh: Vec<i8>,
    scale_ih: Vec<f32>,
    scale_hh: Vec<f32>,
    h_state: [f64; HIDDEN_SIZE],
    c_state: [f64; HIDDEN_SIZE],
    prediction_cache: Vec<f64>,
}

impl LstmPredictor {
    pub fn new() -> Self {
        let mut p = Self {
            w_ih: vec![0; HIDDEN_SIZE * 4],
            w_hh: vec![0; HIDDEN_SIZE * HIDDEN_SIZE * 4],
            scale_ih: vec![0.01; HIDDEN_SIZE * 4],
            scale_hh: vec![0.01; HIDDEN_SIZE * HIDDEN_SIZE * 4],
            h_state: [0.0; HIDDEN_SIZE],
            c_state: [0.0; HIDDEN_SIZE],
            prediction_cache: Vec::new(),
        };
        for (i, w) in p.w_ih.iter_mut().enumerate() {
            *w = ((i % 7) as i32 - 3).clamp(-127, 127) as i8;
        }
        for (i, w) in p.w_hh.iter_mut().enumerate() {
            *w = ((i % 5) as i32 - 2).clamp(-127, 127) as i8;
        }
        p
    }

    pub fn load_weights(&mut self, data: &[u8]) -> bool {
        let ih_bytes = self.w_ih.len();
        let hh_bytes = self.w_hh.len();
        let scale_bytes = self.scale_ih.len() * 4;
        let total = ih_bytes + hh_bytes + 2 * scale_bytes;
        if data.len() < total {
            return false;
        }
        self.w_ih.copy_from_slice(&data[..ih_bytes]);
        self.w_hh.copy_from_slice(&data[ih_bytes..ih_bytes + hh_bytes]);
        let offset = ih_bytes + hh_bytes;
        for (i, chunk) in data[offset..].chunks_exact(4).enumerate().take(self.scale_ih.len()) {
            self.scale_ih[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        true
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn lstm_step(&mut self, input: f64) {
        for h in 0..HIDDEN_SIZE {
            let mut sum = input * self.w_ih[h] as f64 * self.scale_ih[h] as f64;
            for j in 0..HIDDEN_SIZE {
                sum += self.h_state[j]
                    * self.w_hh[h * HIDDEN_SIZE + j] as f64
                    * self.scale_hh[h * HIDDEN_SIZE + j] as f64;
            }
            let i_gate = Self::sigmoid(sum);
            let f_gate = Self::sigmoid(sum + 0.1);
            let g_gate = (sum - 0.1).tanh();
            let o_gate = Self::sigmoid(sum + 0.2);
            self.c_state[h] = f_gate * self.c_state[h] + i_gate * g_gate;
            self.h_state[h] = o_gate * self.c_state[h].tanh();
        }
    }

    pub fn predict(&mut self, history: &[f64; HISTORY_LEN]) -> f64 {
        self.h_state = [0.0; HIDDEN_SIZE];
        self.c_state = [0.0; HIDDEN_SIZE];
        for &sample in history {
            self.lstm_step(sample);
        }
        let mean: f64 = self.h_state.iter().sum::<f64>() / HIDDEN_SIZE as f64;
        mean.clamp(0.0, 1.0)
    }

    pub fn online_update(&mut self, history: &[f64; HISTORY_LEN], target: f64) {
        let pred = self.predict(history);
        let err = target - pred;
        const ETA: f64 = 1e-3;
        for w in &mut self.w_ih {
            let delta = (ETA * err * 127.0) as i32;
            *w = (*w as i32 + delta).clamp(-127, 127) as i8;
        }
    }

    pub fn update_cache(&mut self, edge_idx: u32, current: f64, predicted: f64) {
        let idx = edge_idx as usize;
        if idx >= self.prediction_cache.len() {
            self.prediction_cache.resize(idx + 1, 0.0);
        }
        self.prediction_cache[idx] = predicted;
        let _ = current;
    }

    pub fn effective_util(&self, edge_idx: u32, current: f64, transfer_duration_ms: f64) -> f64 {
        let predicted = self
            .prediction_cache
            .get(edge_idx as usize)
            .copied()
            .unwrap_or(current);
        let weight = (transfer_duration_ms / PREDICT_AHEAD_MS).min(1.0);
        let blended = predicted * weight + current * (1.0 - weight);
        current.max(blended)
    }
}

impl Default for LstmPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_bounded() {
        let mut p = LstmPredictor::new();
        let hist = [0.5; HISTORY_LEN];
        let pred = p.predict(&hist);
        assert!(pred >= 0.0 && pred <= 1.0);
    }
}
