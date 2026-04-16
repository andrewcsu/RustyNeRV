use std::collections::HashMap;
use tch::{Kind, Tensor};
use tch::nn;

/// Per-tensor (axis=-1), per-row (axis=0), or per-column (axis=1) uniform quantization.
/// Returns (quantized_indices, dequantized_tensor).
pub fn quantize_per_tensor(t: &Tensor, bit: i64, axis: i64) -> (Tensor, Tensor) {
    let levels = 2_f64.powi(bit as i32);

    match axis {
        -1 => {
            let valid = t.ne(0.0);
            let valid_vals = t.masked_select(&valid);
            let t_min = valid_vals.min();
            let t_max = valid_vals.max();
            let scale = (&t_max - &t_min) / levels;
            let quant_t = ((t - &t_min) / (&scale + 1e-19)).round();
            let new_t = &t_min + &scale * &quant_t;
            (quant_t, new_t)
        }
        0 => {
            let rows = t.size()[0];
            let mut min_list = Vec::new();
            let mut max_list = Vec::new();
            for i in 0..rows {
                let row = t.get(i);
                let valid = row.ne(0.0);
                if bool::try_from(valid.any()).unwrap_or(false) {
                    let vv = row.masked_select(&valid);
                    min_list.push(f64::try_from(vv.min()).unwrap());
                    max_list.push(f64::try_from(vv.max()).unwrap());
                } else {
                    min_list.push(0.0);
                    max_list.push(0.0);
                }
            }
            let min_t = Tensor::from_slice(&min_list).to_device(t.device());
            let max_t = Tensor::from_slice(&max_list).to_device(t.device());
            let scale = (&max_t - &min_t) / levels;

            let (scale_shaped, min_shaped) = if t.dim() == 4 {
                (
                    scale.view([-1, 1, 1, 1]),
                    min_t.view([-1, 1, 1, 1]),
                )
            } else {
                (scale.view([-1, 1]), min_t.view([-1, 1]))
            };

            let quant_t = ((t - &min_shaped) / (&scale_shaped + 1e-19)).round();
            let new_t = &min_shaped + &scale_shaped * &quant_t;
            (quant_t, new_t)
        }
        1 => {
            let cols = t.size()[1];
            let mut min_list = Vec::new();
            let mut max_list = Vec::new();
            for i in 0..cols {
                let col = t.select(1, i);
                let valid = col.ne(0.0);
                if bool::try_from(valid.any()).unwrap_or(false) {
                    let vv = col.masked_select(&valid);
                    min_list.push(f64::try_from(vv.min()).unwrap());
                    max_list.push(f64::try_from(vv.max()).unwrap());
                } else {
                    min_list.push(0.0);
                    max_list.push(0.0);
                }
            }
            let min_t = Tensor::from_slice(&min_list).to_device(t.device());
            let max_t = Tensor::from_slice(&max_list).to_device(t.device());
            let scale = (&max_t - &min_t) / levels;

            let (scale_shaped, min_shaped) = if t.dim() == 4 {
                (
                    scale.view([1, -1, 1, 1]),
                    min_t.view([1, -1, 1, 1]),
                )
            } else {
                (scale.view([1, -1]), min_t.view([1, -1]))
            };

            let quant_t = ((t - &min_shaped) / (&scale_shaped + 1e-19)).round();
            let new_t = &min_shaped + &scale_shaped * &quant_t;
            (quant_t, new_t)
        }
        _ => panic!("Unsupported quantization axis: {}", axis),
    }
}

/// Perform model quantization: quantize all weights, replace in state_dict,
/// then compute Huffman entropy estimate.
/// Returns encoding_efficiency.
pub fn quantize_and_huffman(
    vs: &mut nn::VarStore,
    quant_bit: i64,
    quant_axis: i64,
) -> f64 {
    let variables = vs.variables();
    let mut all_quant_symbols: Vec<i64> = Vec::new();
    let mut updates: Vec<(String, Tensor)> = Vec::new();

    for (name, tensor) in &variables {
        let large_tf = (tensor.dim() == 2 || tensor.dim() == 4) && !name.contains("bias");
        let axis = if large_tf { quant_axis } else { -1 };

        let (quant_v, new_v) = quantize_per_tensor(tensor, quant_bit, axis);

        let valid = tensor.ne(0.0);
        let valid_quant = quant_v
            .masked_select(&valid)
            .flatten(0, -1)
            .to_kind(Kind::Int64);
        let valid_vec: Vec<i64> = Vec::<i64>::try_from(valid_quant).unwrap_or_default();
        all_quant_symbols.extend(valid_vec);
        updates.push((name.clone(), new_v));
    }

    // Replace weights with dequantized versions via no_grad
    let mut cur_variables = vs.variables();
    for (name, new_val) in &updates {
        if let Some((_, ref mut param)) = cur_variables.iter_mut().find(|(n, _)| n.as_str() == name.as_str()) {
            tch::no_grad(|| {
                param.copy_(new_val);
            });
        }
    }

    // Huffman entropy estimation via Shannon entropy
    let mut freq_map: HashMap<i64, usize> = HashMap::new();
    for &sym in &all_quant_symbols {
        *freq_map.entry(sym).or_insert(0) += 1;
    }

    let total_symbols = all_quant_symbols.len() as f64;
    if total_symbols == 0.0 {
        return 0.0;
    }

    let entropy: f64 = freq_map
        .values()
        .map(|&count| {
            let p = count as f64 / total_symbols;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum();

    let encoding_efficiency = entropy / quant_bit as f64;
    println!(
        "Entropy encoding efficiency for bit {}: {:.4}",
        quant_bit, encoding_efficiency
    );
    encoding_efficiency
}
