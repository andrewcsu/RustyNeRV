use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "nerv-rust", about = "NeRV: Neural Representations for Videos (Rust)")]
pub struct Args {
    // ── Dataset parameters ──────────────────────────────────────────────
    #[arg(long, num_args = 1.., default_values_t = vec![-1])]
    pub vid: Vec<i64>,

    #[arg(long, default_value_t = 1)]
    pub scale: i64,

    #[arg(long, default_value_t = 1)]
    pub frame_gap: usize,

    #[arg(long, default_value_t = 0)]
    pub augment: i64,

    #[arg(long, default_value = "UVG")]
    pub dataset: String,

    #[arg(long, default_value_t = 1)]
    pub test_gap: usize,

    // ── NeRV architecture ───────────────────────────────────────────────
    #[arg(long, default_value = "1.25_80")]
    pub embed: String,

    #[arg(long, default_value = "1024_1")]
    pub stem_dim_num: String,

    #[arg(long, default_value = "9_16_128")]
    pub fc_hw_dim: String,

    #[arg(long, default_value_t = 8.0)]
    pub expansion: f64,

    #[arg(long, default_value_t = 2)]
    pub reduction: i64,

    #[arg(long, num_args = 1.., default_values_t = vec![5, 3, 2, 2, 2])]
    pub strides: Vec<i64>,

    #[arg(long, default_value_t = 1)]
    pub num_blocks: i64,

    #[arg(long, default_value = "none")]
    pub norm: String,

    #[arg(long, default_value = "gelu")]
    pub act: String,

    #[arg(long, default_value_t = 32)]
    pub lower_width: i64,

    #[arg(long, default_value_t = false)]
    pub single_res: bool,

    #[arg(long, default_value = "conv")]
    pub conv_type: String,

    // ── Training parameters ─────────────────────────────────────────────
    #[arg(short = 'j', long, default_value_t = 4)]
    pub workers: usize,

    #[arg(short = 'b', long = "batchSize", default_value_t = 1)]
    pub batch_size: i64,

    #[arg(long, default_value_t = false)]
    pub not_resume_epoch: bool,

    #[arg(short = 'e', long, default_value_t = 150)]
    pub epochs: i64,

    #[arg(long, default_value_t = 1)]
    pub cycles: i64,

    #[arg(long, default_value_t = 0.2)]
    pub warmup: f64,

    #[arg(long, default_value_t = 0.001)]
    pub lr: f64,

    #[arg(long, default_value = "cosine")]
    pub lr_type: String,

    #[arg(long, num_args = 0..)]
    pub lr_steps: Vec<f64>,

    #[arg(long, default_value_t = 0.5)]
    pub beta: f64,

    #[arg(long, default_value = "L2")]
    pub loss_type: String,

    #[arg(long, default_value_t = 1.0)]
    pub lw: f64,

    #[arg(long, default_value_t = false)]
    pub sigmoid: bool,

    // ── Evaluation parameters ───────────────────────────────────────────
    #[arg(long, default_value_t = false)]
    pub eval_only: bool,

    #[arg(long, default_value_t = 50)]
    pub eval_freq: i64,

    #[arg(long, default_value_t = -1)]
    pub quant_bit: i64,

    #[arg(long, default_value_t = 0)]
    pub quant_axis: i64,

    #[arg(long, default_value_t = false)]
    pub dump_images: bool,

    #[arg(long, default_value_t = false)]
    pub eval_fps: bool,

    // ── Pruning parameters ──────────────────────────────────────────────
    #[arg(long, num_args = 1.., default_values_t = vec![0.0])]
    pub prune_steps: Vec<f64>,

    #[arg(long, default_value_t = 1.0)]
    pub prune_ratio: f64,

    // ── Misc / logging ──────────────────────────────────────────────────
    #[arg(long = "manualSeed", default_value_t = 1)]
    pub manual_seed: i64,

    #[arg(long, default_value_t = false)]
    pub debug: bool,

    #[arg(short = 'p', long = "print-freq", default_value_t = 50)]
    pub print_freq: usize,

    #[arg(long, default_value = "None")]
    pub weight: String,

    #[arg(long, default_value_t = false)]
    pub overwrite: bool,

    #[arg(long, default_value = "unify")]
    pub outf: String,

    #[arg(long, default_value = "")]
    pub suffix: String,
}

impl Args {
    /// Derived fields computed after parsing, matching the Python post-processing.
    pub fn warmup_epochs(&self) -> i64 {
        (self.warmup * self.epochs as f64) as i64
    }

    pub fn output_dir(&self) -> String {
        let base = if self.debug {
            "output/debug".to_string()
        } else {
            format!("output/{}", self.outf)
        };
        let exp_id = self.exp_id();
        format!("{}/{}", base, exp_id)
    }

    pub fn exp_id(&self) -> String {
        let prune_str = if self.prune_ratio < 1.0 && !self.eval_only {
            let steps: Vec<String> = self.prune_steps.iter().map(|x| x.to_string()).collect();
            format!("_Prune{}_{}", self.prune_ratio, steps.join(","))
        } else {
            String::new()
        };

        let strides_str: Vec<String> = self.strides.iter().map(|x| x.to_string()).collect();
        let res_str = if self.single_res {
            "Sin".to_string()
        } else {
            format!("_lw{}_multi", self.lw)
        };
        let eval_str = if self.eval_only { "_eval" } else { "" };
        let extra_str = format!("_Strd{}_{}Res{}", strides_str.join(","), res_str, eval_str);

        let norm_str = if self.norm == "none" { "" } else { &self.norm };
        let warmup_epochs = self.warmup_epochs();

        format!(
            "{dataset}/embed{embed}_{stem}_fc_{fc}__exp{exp}_reduce{red}_low{low}_blk{blk}_cycle{cyc}\
             _gap{gap}_e{epochs}_warm{warm}_b{bs}_{conv}_lr{lr}_{lrt}\
             _{loss}{norm}{extra}{prune}_act{act}_{suffix}",
            dataset = self.dataset,
            embed = self.embed,
            stem = self.stem_dim_num,
            fc = self.fc_hw_dim,
            exp = self.expansion,
            red = self.reduction,
            low = self.lower_width,
            blk = self.num_blocks,
            cyc = self.cycles,
            gap = self.frame_gap,
            epochs = self.epochs,
            warm = warmup_epochs,
            bs = self.batch_size,
            conv = self.conv_type,
            lr = self.lr,
            lrt = self.lr_type,
            loss = self.loss_type,
            norm = norm_str,
            extra = extra_str,
            prune = prune_str,
            act = self.act,
            suffix = self.suffix,
        )
    }

    pub fn vid_list(&self) -> Option<Vec<usize>> {
        if self.vid.len() == 1 && self.vid[0] == -1 {
            None
        } else {
            Some(self.vid.iter().map(|&v| v as usize).collect())
        }
    }
}
