use std::fs;
use std::time::Instant;

use tch::{no_grad, Device, Kind, Tensor};
use tch::nn;

use crate::config::Args;
use crate::data::DataLoader;
use crate::encoding::PositionalEncoding;
use crate::metrics::{msssim_fn, psnr_fn, round_tensor};
use crate::model::Generator;
use crate::quantize;

/// Run evaluation. Returns (psnr_per_stage, msssim_per_stage).
pub fn evaluate(
    model: &Generator,
    val_loader: &mut DataLoader,
    pe: &PositionalEncoding,
    device: Device,
    args: &Args,
    vs: &mut nn::VarStore,
    outf: &str,
) -> (Tensor, Tensor) {
    // Optional quantization
    if args.quant_bit != -1 {
        let efficiency = quantize::quantize_and_huffman(vs, args.quant_bit, args.quant_axis);
        let print_str = format!(
            "Entropy encoding efficiency for bit {}: {:.4}",
            args.quant_bit, efficiency
        );
        let eval_path = format!("{}/eval.txt", outf);
        append_to_file(&eval_path, &print_str);
    }

    // Optional image dump setup
    let visual_dir = format!("{}/visualize", outf);
    if args.dump_images {
        println!("Saving predictions to {}", visual_dir);
        fs::create_dir_all(&visual_dir).ok();
    }

    let mut psnr_list: Vec<Tensor> = Vec::new();
    let mut msssim_list: Vec<Tensor> = Vec::new();
    let mut time_list: Vec<f64> = Vec::new();
    let fwd_num = if args.eval_fps { 10 } else { 1 };

    val_loader.reset();
    let total_batches = val_loader.len();

    for (i, (data, norm_idx)) in val_loader.enumerate() {
        if i > 10 && args.debug {
            break;
        }
        let data = data.to_device(device);
        let embed_input = pe.forward(&norm_idx.to_device(device));

        let output_list = no_grad(|| {
            let mut last_output = Vec::new();
            for _ in 0..fwd_num {
                let start = Instant::now();
                let out = model.forward_t(&embed_input, false);
                if device != Device::Cpu {
                    tch::Cuda::synchronize(0);
                }
                time_list.push(start.elapsed().as_secs_f64());
                last_output = out;
            }
            last_output
        });

        // Dump images
        if args.dump_images {
            let bs = args.batch_size as usize;
            for batch_ind in 0..bs.min(output_list.last().unwrap().size()[0] as usize) {
                let full_ind = i * bs + batch_ind;
                save_tensor_as_png(
                    &output_list.last().unwrap().get(batch_ind as i64),
                    &format!("{}/pred_{}.png", visual_dir, full_ind),
                );
                save_tensor_as_png(
                    &data.get(batch_ind as i64),
                    &format!("{}/gt_{}.png", visual_dir, full_ind),
                );
            }
        }

        // Compute metrics
        let target_list: Vec<Tensor> = output_list
            .iter()
            .map(|x| {
                let h = x.size()[2];
                let w = x.size()[3];
                data.adaptive_avg_pool2d(&[h, w])
            })
            .collect();

        psnr_list.push(no_grad(|| psnr_fn(&output_list, &target_list)));
        msssim_list.push(no_grad(|| msssim_fn(&output_list, &target_list)));

        let val_psnr = Tensor::cat(&psnr_list, 0).mean_dim(0, false, Kind::Float);
        let val_msssim = Tensor::cat(&msssim_list, 0)
            .to_kind(Kind::Float)
            .mean_dim(0, false, Kind::Float);

        if i % args.print_freq == 0 {
            let fps = fwd_num as f64 * (i + 1) as f64 * args.batch_size as f64
                / time_list.iter().sum::<f64>();
            let print_str = format!(
                "Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {:.2}",
                i + 1,
                total_batches,
                round_tensor(&val_psnr, 2),
                round_tensor(&val_msssim, 4),
                fps,
            );
            println!("{}", print_str);
            let log_path = format!("{}/rank0.txt", outf);
            append_to_file(&log_path, &print_str);
        }
    }

    let val_psnr = Tensor::cat(&psnr_list, 0).mean_dim(0, false, Kind::Float);
    let val_msssim = Tensor::cat(&msssim_list, 0)
        .to_kind(Kind::Float)
        .mean_dim(0, false, Kind::Float);

    (val_psnr, val_msssim)
}

fn save_tensor_as_png(tensor: &Tensor, path: &str) {
    // tensor shape: [C, H, W] in [0,1] float
    let t = (tensor.clamp(0.0, 1.0) * 255.0)
        .to_kind(Kind::Uint8)
        .to_device(Device::Cpu);
    let c = t.size()[0] as u32;
    let h = t.size()[1] as u32;
    let w = t.size()[2] as u32;
    // CHW -> HWC
    let hwc = t.permute(&[1, 2, 0]).contiguous();
    let data: Vec<u8> = Vec::<u8>::try_from(hwc.flatten(0, -1)).unwrap_or_default();
    if c == 3 {
        let img = image::RgbImage::from_raw(w, h, data).expect("failed to create image");
        img.save(path).expect("failed to save image");
    }
}

fn append_to_file(path: &str, content: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        writeln!(f, "{}", content).ok();
    }
}
