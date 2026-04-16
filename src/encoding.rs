use tch::Tensor;

pub struct PositionalEncoding {
    pub embed_length: i64,
    lbase: f64,
    levels: i64,
    is_none: bool,
}

impl PositionalEncoding {
    pub fn new(pe_embed: &str) -> Self {
        let pe_lower = pe_embed.to_lowercase();
        if pe_lower == "none" {
            return Self {
                embed_length: 1,
                lbase: 0.0,
                levels: 0,
                is_none: true,
            };
        }
        let parts: Vec<&str> = pe_embed.split('_').collect();
        let lbase: f64 = parts[0].parse().expect("invalid PE base");
        let levels: i64 = parts[1].parse::<f64>().expect("invalid PE levels") as i64;
        Self {
            embed_length: 2 * levels,
            lbase,
            levels,
            is_none: false,
        }
    }

    /// pos: Tensor of shape [B] (frame indices in [0, 1])
    /// returns: Tensor of shape [B, embed_length]
    pub fn forward(&self, pos: &Tensor) -> Tensor {
        if self.is_none {
            return pos.unsqueeze(-1);
        }
        let mut pe_list: Vec<Tensor> = Vec::with_capacity(2 * self.levels as usize);
        for i in 0..self.levels {
            let scale = self.lbase.powi(i as i32) * std::f64::consts::PI;
            let temp = pos * scale;
            pe_list.push(temp.sin());
            pe_list.push(temp.cos());
        }
        Tensor::stack(&pe_list, 1)
    }
}
