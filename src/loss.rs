use tch::{Kind, Tensor};

use crate::ssim;

pub fn loss_fn(pred: &Tensor, target: &Tensor, loss_type: &str) -> Tensor {
    let target = target.detach();
    match loss_type {
        "L2" => pred.mse_loss(&target, tch::Reduction::Mean),
        "L1" => (pred - &target).abs().mean(Kind::Float),
        "SSIM" => 1.0 - ssim::ssim(pred, &target, 1.0, true),
        "Fusion1" => {
            pred.mse_loss(&target, tch::Reduction::Mean) * 0.3
                + (1.0 - ssim::ssim(pred, &target, 1.0, true)) * 0.7
        }
        "Fusion2" => {
            (pred - &target).abs().mean(Kind::Float) * 0.3
                + (1.0 - ssim::ssim(pred, &target, 1.0, true)) * 0.7
        }
        "Fusion3" => {
            pred.mse_loss(&target, tch::Reduction::Mean) * 0.5
                + (1.0 - ssim::ssim(pred, &target, 1.0, true)) * 0.5
        }
        "Fusion4" => {
            (pred - &target).abs().mean(Kind::Float) * 0.5
                + (1.0 - ssim::ssim(pred, &target, 1.0, true)) * 0.5
        }
        "Fusion5" => {
            pred.mse_loss(&target, tch::Reduction::Mean) * 0.7
                + (1.0 - ssim::ssim(pred, &target, 1.0, true)) * 0.3
        }
        "Fusion6" => {
            (pred - &target).abs().mean(Kind::Float) * 0.7
                + (1.0 - ssim::ssim(pred, &target, 1.0, true)) * 0.3
        }
        "Fusion7" => {
            pred.mse_loss(&target, tch::Reduction::Mean) * 0.7
                + (pred - &target).abs().mean(Kind::Float) * 0.3
        }
        "Fusion8" => {
            pred.mse_loss(&target, tch::Reduction::Mean) * 0.5
                + (pred - &target).abs().mean(Kind::Float) * 0.5
        }
        "Fusion9" => {
            (pred - &target).abs().mean(Kind::Float) * 0.9
                + (1.0 - ssim::ssim(pred, &target, 1.0, true)) * 0.1
        }
        "Fusion10" => {
            (pred - &target).abs().mean(Kind::Float) * 0.7
                + (1.0 - ssim::ms_ssim(pred, &target, 1.0, true)) * 0.3
        }
        "Fusion11" => {
            (pred - &target).abs().mean(Kind::Float) * 0.9
                + (1.0 - ssim::ms_ssim(pred, &target, 1.0, true)) * 0.1
        }
        "Fusion12" => {
            (pred - &target).abs().mean(Kind::Float) * 0.8
                + (1.0 - ssim::ms_ssim(pred, &target, 1.0, true)) * 0.2
        }
        _ => panic!("Unknown loss type: {}", loss_type),
    }
}
