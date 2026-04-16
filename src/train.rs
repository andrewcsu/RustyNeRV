use std::fs;
use std::time::Instant;

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::config::Args;
use crate::data::{CustomDataSet, DataLoader};
use crate::encoding::PositionalEncoding;
use crate::eval;
use crate::loss::loss_fn;
use crate::metrics::{msssim_fn, psnr_fn, round_tensor};
use crate::model::{count_parameters, Generator};
use crate::pruning;
use crate::scheduler::adjust_lr;

pub fn train(args: &Args) -> anyhow::Result<()> {
    tch::manual_seed(args.manual_seed);
    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        println!("CUDA not available, using CPU");
        Device::Cpu
    };

    let outf = args.output_dir();
    fs::create_dir_all(&outf)?;

    // ── Build model ─────────────────────────────────────────────────────
    let pe = PositionalEncoding::new(&args.embed);
    let embed_length = pe.embed_length;

    let mut vs = nn::VarStore::new(device);
    let model = Generator::new(
        &vs.root(),
        embed_length,
        &args.stem_dim_num,
        &args.fc_hw_dim,
        args.expansion,
        args.num_blocks,
        &args.norm,
        &args.act,
        true,
        args.reduction,
        &args.conv_type,
        &args.strides,
        args.single_res,
        args.lower_width,
        args.sigmoid,
    );

    let total_params = count_parameters(&vs);
    println!("Model Params: {:.4}M", total_params);

    {
        let log_path = format!("{}/rank0.txt", outf);
        append_to_file(&log_path, &format!("Params: {:.4}M", total_params));
    }

    // ── Pruning setup ───────────────────────────────────────────────────
    let prune_net = args.prune_ratio < 1.0;
    let prune_base_ratio = if prune_net {
        args.prune_ratio.powf(1.0 / args.prune_steps.len() as f64)
    } else {
        1.0
    };
    let prune_step_epochs: Vec<i64> = args
        .prune_steps
        .iter()
        .map(|&x| (x * args.epochs as f64) as i64)
        .collect();
    let mut prune_num: u32 = 0;

    // ── Optimizer ───────────────────────────────────────────────────────
    let mut optimizer = nn::Adam {
        beta1: args.beta,
        beta2: 0.999,
        wd: 0.0,
        eps: 1e-8,
        amsgrad: false,
    }
    .build(&vs, args.lr)?;

    // ── Load checkpoint ─────────────────────────────────────────────────
    let mut start_epoch: i64 = 0;
    let mut train_best_psnr: f64 = 0.0;
    let mut train_best_msssim: f64 = 0.0;
    let mut val_best_psnr: f64 = 0.0;
    let mut val_best_msssim: f64 = 0.0;

    if args.weight != "None" {
        println!("Loading checkpoint '{}'", args.weight);
        vs.load(&args.weight)?;
        println!("Loaded checkpoint '{}'", args.weight);
    }

    let latest_path = format!("{}/model_latest.pth", outf);
    if std::path::Path::new(&latest_path).exists() {
        println!("Auto-resuming from '{}'", latest_path);
        vs.load(&latest_path)?;
        // Note: epoch counter would ideally be saved/loaded from a sidecar file.
        // For simplicity we start from 0 unless --not_resume_epoch is unset.
    }

    if args.not_resume_epoch {
        start_epoch = 0;
    }

    // ── Dataset ─────────────────────────────────────────────────────────
    let data_dir = format!("./data/{}", args.dataset.to_lowercase());
    let train_dataset = CustomDataSet::new(&data_dir, args.vid_list().as_deref(), args.frame_gap)?;
    let val_dataset = CustomDataSet::new(&data_dir, args.vid_list().as_deref(), args.test_gap)?;

    let data_size = train_dataset.len();
    println!("Dataset size: {} frames", data_size);

    // ── Eval-only mode ──────────────────────────────────────────────────
    if args.eval_only {
        println!("Evaluation ...");
        let mut val_loader = DataLoader::new(val_dataset, args.batch_size as usize, false);
        let (val_psnr, val_msssim) =
            eval::evaluate(&model, &mut val_loader, &pe, device, args, &mut vs, &outf);
        let print_str = format!(
            "PSNR/ms_ssim on validate set for bit {} with axis {}: {}/{}",
            args.quant_bit,
            args.quant_axis,
            round_tensor(&val_psnr, 2),
            round_tensor(&val_msssim, 4),
        );
        println!("{}", print_str);
        let eval_path = format!("{}/eval.txt", outf);
        append_to_file(&eval_path, &print_str);
        return Ok(());
    }

    // ── Training loop ───────────────────────────────────────────────────
    let total_epochs = args.epochs * args.cycles;
    let warmup_epochs = args.warmup_epochs();
    let global_start = Instant::now();

    for epoch in start_epoch..total_epochs {
        let epoch_start = Instant::now();

        // ── Prune if scheduled ──────────────────────────────────────────
        if prune_net && prune_step_epochs.contains(&epoch) {
            prune_num += 1;
            let amount = 1.0 - prune_base_ratio.powi(prune_num as i32);
            pruning::global_l1_unstructured_prune(&mut vs, amount);
            let sparsity = pruning::compute_sparsity(&vs);
            println!("Model sparsity at Epoch{}: {:.4}", epoch, sparsity);
        }

        let mut psnr_list: Vec<Tensor> = Vec::new();
        let mut msssim_list: Vec<Tensor> = Vec::new();
        let train_loader =
            DataLoader::new(
                CustomDataSet::new(&data_dir, args.vid_list().as_deref(), args.frame_gap)?,
                args.batch_size as usize,
                true,
            );
        let num_batches = train_loader.len();

        for (i, (data, norm_idx)) in train_loader.enumerate() {
            if i > 10 && args.debug {
                break;
            }
            let data = data.to_device(device);
            let embed_input = pe.forward(&norm_idx.to_device(device));

            let output_list = model.forward_t(&embed_input, true);

            // Multi-scale targets via adaptive average pooling
            let target_list: Vec<Tensor> = output_list
                .iter()
                .map(|x| {
                    let h = x.size()[2];
                    let w = x.size()[3];
                    data.adaptive_avg_pool2d(&[h, w])
                })
                .collect();

            // Compute weighted loss across stages
            let num_stages = output_list.len();
            let mut loss_sum = Tensor::zeros(&[], (Kind::Float, device));
            for (stage_idx, (output, target)) in
                output_list.iter().zip(target_list.iter()).enumerate()
            {
                let stage_loss = loss_fn(output, target, &args.loss_type);
                let weight = if stage_idx < num_stages - 1 {
                    args.lw
                } else {
                    1.0
                };
                loss_sum = loss_sum + stage_loss * weight;
            }

            let effective_epoch = epoch % args.epochs;
            let lr = adjust_lr(
                &mut optimizer,
                effective_epoch,
                i,
                data_size,
                args.lr,
                warmup_epochs,
                args.epochs,
                &args.lr_type,
                &args.lr_steps,
            );

            optimizer.zero_grad();
            loss_sum.backward();
            optimizer.step();

            // Compute metrics
            psnr_list.push(tch::no_grad(|| psnr_fn(&output_list, &target_list)));
            msssim_list.push(tch::no_grad(|| msssim_fn(&output_list, &target_list)));

            if i % args.print_freq == 0 || i == num_batches - 1 {
                let train_psnr = Tensor::cat(&psnr_list, 0).mean_dim(0, false, Kind::Float);
                let train_msssim = Tensor::cat(&msssim_list, 0)
                    .to_kind(Kind::Float)
                    .mean_dim(0, false, Kind::Float);
                let time_str = chrono::Local::now().format("%Y/%m/%d %H:%M:%S").to_string();
                let print_str = format!(
                    "[{}] Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}",
                    time_str,
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    num_batches,
                    lr,
                    round_tensor(&train_psnr, 2),
                    round_tensor(&train_msssim, 4),
                );
                println!("{}", print_str);
                let log_path = format!("{}/rank0.txt", outf);
                append_to_file(&log_path, &print_str);
            }
        }

        // Epoch summary
        let train_psnr = Tensor::cat(&psnr_list, 0).mean_dim(0, false, Kind::Float);
        let train_msssim = Tensor::cat(&msssim_list, 0)
            .to_kind(Kind::Float)
            .mean_dim(0, false, Kind::Float);

        let last_psnr: f64 = f64::try_from(train_psnr.get(-1)).unwrap_or(0.0);
        let last_msssim: f64 = f64::try_from(train_msssim.get(-1)).unwrap_or(0.0);
        let is_train_best = last_psnr > train_best_psnr;
        if is_train_best {
            train_best_psnr = last_psnr;
        }
        if last_msssim > train_best_msssim {
            train_best_msssim = last_msssim;
        }

        let epoch_secs = epoch_start.elapsed().as_secs_f64();
        let avg_secs =
            global_start.elapsed().as_secs_f64() / (epoch - start_epoch + 1) as f64;
        let print_str = format!(
            "\tcurrent: {:.2}\t best: {:.2}\t msssim_best: {:.4}\t Time/epoch: Current:{:.2} Average:{:.2}",
            last_psnr, train_best_psnr, train_best_msssim, epoch_secs, avg_secs,
        );
        println!("{}", print_str);
        append_to_file(&format!("{}/rank0.txt", outf), &print_str);

        // Save latest checkpoint
        vs.save(format!("{}/model_latest.pth", outf))?;
        if is_train_best {
            vs.save(format!("{}/model_train_best.pth", outf))?;
        }

        // ── Periodic validation ─────────────────────────────────────────
        if (epoch + 1) % args.eval_freq == 0 || epoch > total_epochs - 10 {
            let val_start = Instant::now();
            let mut val_loader =
                DataLoader::new(
                    CustomDataSet::new(&data_dir, args.vid_list().as_deref(), args.test_gap)?,
                    args.batch_size as usize,
                    false,
                );
            let (val_psnr, val_msssim) =
                eval::evaluate(&model, &mut val_loader, &pe, device, args, &mut vs, &outf);
            let val_secs = val_start.elapsed().as_secs_f64();

            let val_last_psnr: f64 = f64::try_from(val_psnr.get(-1)).unwrap_or(0.0);
            let val_last_msssim: f64 = f64::try_from(val_msssim.get(-1)).unwrap_or(0.0);
            let is_val_best = val_last_psnr > val_best_psnr;
            if is_val_best {
                val_best_psnr = val_last_psnr;
            }
            if val_last_msssim > val_best_msssim {
                val_best_msssim = val_last_msssim;
            }

            let print_str = format!(
                "Eval best_PSNR at epoch{}: current: {:.2}\tbest: {:.2}\tbest_msssim: {:.4}\tTime: {:.2}s",
                epoch + 1, val_last_psnr, val_best_psnr, val_best_msssim, val_secs,
            );
            println!("{}", print_str);
            append_to_file(&format!("{}/rank0.txt", outf), &print_str);

            if is_val_best {
                vs.save(format!("{}/model_val_best.pth", outf))?;
            }
        }
    }

    println!(
        "Training complete in: {:.2}s",
        global_start.elapsed().as_secs_f64()
    );
    Ok(())
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
