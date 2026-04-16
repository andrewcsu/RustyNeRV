use tch::Kind;
use tch::nn;

/// Identify which variable names in the VarStore are prunable.
/// Matches the Python logic: stem weights and layers.*.conv.conv.weight.
pub fn prunable_param_names(vs: &nn::VarStore) -> Vec<String> {
    vs.variables()
        .into_iter()
        .filter_map(|(name, _)| {
            if !name.contains("weight") {
                return None;
            }
            if name.contains("stem") || (name.starts_with("layers.") && name.contains("conv")) {
                Some(name)
            } else {
                None
            }
        })
        .collect()
}

/// Apply global L1 unstructured pruning.
/// Zeros out the smallest-magnitude weights across all prunable tensors
/// until `amount` fraction of total prunable weights are zero.
pub fn global_l1_unstructured_prune(vs: &mut nn::VarStore, amount: f64) {
    let names = prunable_param_names(vs);
    if names.is_empty() || amount <= 0.0 {
        return;
    }

    let variables = vs.variables();
    let mut all_abs: Vec<f32> = Vec::new();
    let mut sizes: Vec<(String, i64)> = Vec::new();

    for name in &names {
        if let Some((_, tensor)) = variables.iter().find(|(n, _)| n.as_str() == name.as_str()) {
            let abs_tensor = tensor.abs().flatten(0, -1).to_kind(Kind::Float);
            let numel = abs_tensor.size()[0];
            let flat: Vec<f32> = Vec::<f32>::try_from(abs_tensor).unwrap_or_default();
            sizes.push((name.clone(), numel));
            all_abs.extend(flat);
        }
    }

    let total = all_abs.len();
    let num_to_prune = ((total as f64) * amount).round() as usize;
    if num_to_prune == 0 {
        return;
    }

    let mut sorted = all_abs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = sorted[num_to_prune.min(total - 1)];

    // Apply masks: zero out weights at or below threshold
    let mut variables = vs.variables();
    for (name, _size) in &sizes {
        if let Some((_, ref mut tensor)) = variables.iter_mut().find(|(n, _)| n.as_str() == name.as_str()) {
            let mask = tensor.abs().gt(threshold as f64);
            let kind = tensor.kind();
            let masked = tensor.shallow_clone() * mask.to_kind(kind);
            tch::no_grad(|| {
                tensor.copy_(&masked);
            });
        }
    }
}

/// Compute and return the sparsity ratio for prunable params.
pub fn compute_sparsity(vs: &nn::VarStore) -> f64 {
    let names = prunable_param_names(vs);
    let variables = vs.variables();
    let mut total: i64 = 0;
    let mut zeros: i64 = 0;

    for name in &names {
        if let Some((_, tensor)) = variables.iter().find(|(n, _)| n.as_str() == name.as_str()) {
            let n = tensor.numel() as i64;
            let z: i64 = i64::try_from(tensor.eq(0.0).sum(Kind::Int64)).unwrap_or(0);
            total += n;
            zeros += z;
        }
    }

    if total == 0 {
        0.0
    } else {
        zeros as f64 / total as f64
    }
}
