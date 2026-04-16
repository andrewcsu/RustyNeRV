use tch::{Kind, Tensor};

use crate::ssim;

/// Compute PSNR for each stage.
/// Returns tensor of shape [batch_size, num_stages].
pub fn psnr_fn(output_list: &[Tensor], target_list: &[Tensor]) -> Tensor {
    let mut psnr_per_stage: Vec<Tensor> = Vec::new();
    for (output, target) in output_list.iter().zip(target_list.iter()) {
        let l2 = output.detach().mse_loss(&target.detach(), tch::Reduction::Mean);
        let psnr = l2.log10() * (-10.0);
        let bs = output.size()[0];
        let psnr = psnr.view([1, 1]).expand(&[bs, 1], false);
        psnr_per_stage.push(psnr);
    }
    Tensor::cat(&psnr_per_stage, 1)
}

/// Compute MS-SSIM for each stage.
/// Returns tensor of shape [batch_size, num_stages].
/// Only computes MS-SSIM when spatial height >= 160; otherwise returns 0.
pub fn msssim_fn(output_list: &[Tensor], target_list: &[Tensor]) -> Tensor {
    let mut msssim_per_stage: Vec<Tensor> = Vec::new();
    let device = output_list.last().unwrap().device();
    let bs = output_list.last().unwrap().size()[0];

    for (output, target) in output_list.iter().zip(target_list.iter()) {
        let h = output.size()[2];
        let val = if h >= 160 {
            ssim::ms_ssim(
                &output.to_kind(Kind::Float).detach(),
                &target.detach(),
                1.0,
                true,
            )
        } else {
            Tensor::zeros(&[], (Kind::Float, device))
        };
        msssim_per_stage.push(val.view([1]));
    }
    let msssim = Tensor::cat(&msssim_per_stage, 0);
    msssim.view([1, -1]).expand(&[bs, -1], false)
}

/// Format a tensor to a comma-separated string with given decimal places.
pub fn round_tensor(x: &Tensor, decimals: usize) -> String {
    let flat: Vec<f64> = Vec::<f64>::try_from(x.flatten(0, -1).to_kind(Kind::Double)).unwrap_or_default();
    flat.iter()
        .map(|v| format!("{:.prec$}", v, prec = decimals))
        .collect::<Vec<_>>()
        .join(",")
}
