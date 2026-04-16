use tch::{Kind, Tensor};

/// Create a 1-D Gaussian kernel of a given size and sigma.
fn gaussian_1d(size: i64, sigma: f64) -> Tensor {
    let coords = Tensor::arange(size, (Kind::Float, tch::Device::Cpu))
        - (size as f64 - 1.0) / 2.0;
    let g = (-(&coords * &coords) / (2.0 * sigma * sigma)).exp();
    &g / g.sum(Kind::Float)
}

/// Create a 2-D Gaussian window of shape [1, 1, size, size] on the given device.
fn gaussian_window(size: i64, sigma: f64, device: tch::Device) -> Tensor {
    let g1 = gaussian_1d(size, sigma);
    let window_2d = g1.unsqueeze(1).matmul(&g1.unsqueeze(0));
    window_2d.unsqueeze(0).unsqueeze(0).to_device(device)
}

/// Compute SSIM between two image tensors.
/// x, y: [B, C, H, W] in [0, 1].
/// Returns a scalar (mean SSIM over all pixels and channels).
pub fn ssim(x: &Tensor, y: &Tensor, data_range: f64, size_average: bool) -> Tensor {
    let c = x.size()[1];
    let win_size: i64 = 11;
    let sigma = 1.5;
    let window = gaussian_window(win_size, sigma, x.device()).expand(&[c, 1, win_size, win_size], false);

    let k1 = 0.01;
    let k2 = 0.03;
    let c1 = (k1 * data_range) * (k1 * data_range);
    let c2 = (k2 * data_range) * (k2 * data_range);

    let pad = win_size / 2;

    let mu1 = x.conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c);
    let mu2 = y.conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c);

    let mu1_sq = &mu1 * &mu1;
    let mu2_sq = &mu2 * &mu2;
    let mu1_mu2 = &mu1 * &mu2;

    let sigma1_sq = x.pow_tensor_scalar(2).conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c) - &mu1_sq;
    let sigma2_sq = y.pow_tensor_scalar(2).conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c) - &mu2_sq;
    let sigma12 = (x * y).conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c) - &mu1_mu2;

    let numerator = (&mu1_mu2 * 2.0 + c1) * (&sigma12 * 2.0 + c2);
    let denominator = (&mu1_sq + &mu2_sq + c1) * (&sigma1_sq + &sigma2_sq + c2);
    let ssim_map = numerator / denominator;

    if size_average {
        ssim_map.mean(Kind::Float)
    } else {
        ssim_map.mean_dim(&[1i64, 2, 3][..], false, Kind::Float)
    }
}

/// Compute MS-SSIM between two image tensors.
/// x, y: [B, C, H, W] in [0, 1].
/// Returns a scalar if size_average, else [B].
pub fn ms_ssim(x: &Tensor, y: &Tensor, data_range: f64, size_average: bool) -> Tensor {
    let weights: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
    let c = x.size()[1];
    let win_size: i64 = 11;
    let sigma = 1.5;
    let window = gaussian_window(win_size, sigma, x.device()).expand(&[c, 1, win_size, win_size], false);

    let k1 = 0.01;
    let k2 = 0.03;
    let c1 = (k1 * data_range) * (k1 * data_range);
    let c2 = (k2 * data_range) * (k2 * data_range);

    let pad = win_size / 2;
    let levels = weights.len();

    let mut mcs_list: Vec<Tensor> = Vec::new();
    let mut cur_x = x.shallow_clone();
    let mut cur_y = y.shallow_clone();

    for i in 0..levels {
        let mu1 = cur_x.conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c);
        let mu2 = cur_y.conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c);

        let mu1_sq = &mu1 * &mu1;
        let mu2_sq = &mu2 * &mu2;
        let mu1_mu2 = &mu1 * &mu2;

        let sigma1_sq = cur_x.pow_tensor_scalar(2).conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c) - &mu1_sq;
        let sigma2_sq = cur_y.pow_tensor_scalar(2).conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c) - &mu2_sq;
        let sigma12 = (&cur_x * &cur_y).conv2d(&window, None::<&Tensor>, &[1], &[pad], &[1], c) - &mu1_mu2;

        let cs = (&sigma12 * 2.0 + c2) / (&sigma1_sq + &sigma2_sq + c2);

        if i == levels - 1 {
            // Last level: compute full SSIM (luminance * contrast*structure)
            let l = (&mu1_mu2 * 2.0 + c1) / (&mu1_sq + &mu2_sq + c1);
            let ssim_val = l.mean_dim(&[1i64, 2, 3][..], false, Kind::Float)
                * cs.mean_dim(&[1i64, 2, 3][..], false, Kind::Float);
            mcs_list.push(ssim_val);
        } else {
            mcs_list.push(cs.mean_dim(&[1i64, 2, 3][..], false, Kind::Float));
            // Downsample by 2
            let h = cur_x.size()[2] / 2;
            let w = cur_x.size()[3] / 2;
            cur_x = cur_x.adaptive_avg_pool2d(&[h, w]);
            cur_y = cur_y.adaptive_avg_pool2d(&[h, w]);
        }
    }

    // Weighted product across scales
    let mut result = Tensor::ones(&[x.size()[0]], (Kind::Float, x.device()));
    for (i, mcs) in mcs_list.iter().enumerate() {
        let clamped = mcs.clamp(0.0, 1.0);
        result = result * clamped.pow_tensor_scalar(weights[i]);
    }

    if size_average {
        result.mean(Kind::Float)
    } else {
        result
    }
}
