use tch::nn;

/// Adjust learning rate for the current iteration, matching Python adjust_lr.
/// Returns the new learning rate.
pub fn adjust_lr(
    optimizer: &mut nn::Optimizer,
    cur_epoch: i64,
    cur_iter: usize,
    data_size: usize,
    base_lr: f64,
    warmup_epochs: i64,
    total_epochs: i64,
    lr_type: &str,
    lr_steps: &[f64],
) -> f64 {
    let epoch_frac = cur_epoch as f64 + (cur_iter as f64 / data_size as f64);

    let lr_mult = if epoch_frac < warmup_epochs as f64 {
        0.1 + 0.9 * epoch_frac / warmup_epochs as f64
    } else {
        match lr_type {
            "cosine" => {
                let progress =
                    (epoch_frac - warmup_epochs as f64) / (total_epochs - warmup_epochs) as f64;
                0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            "step" => {
                let count = lr_steps
                    .iter()
                    .filter(|&&s| epoch_frac >= s)
                    .count();
                0.1_f64.powi(count as i32)
            }
            "const" | "plateau" => 1.0,
            _ => panic!("Unknown lr_type: {}", lr_type),
        }
    };

    let new_lr = base_lr * lr_mult;
    optimizer.set_lr(new_lr);
    new_lr
}
