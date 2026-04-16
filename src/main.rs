mod config;
mod data;
mod encoding;
mod eval;
mod loss;
mod metrics;
mod model;
mod pruning;
mod quantize;
mod scheduler;
mod ssim;
mod train;

use clap::Parser;
use config::Args;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("{:?}", args);

    if args.overwrite {
        let outf = args.output_dir();
        if std::path::Path::new(&outf).is_dir() {
            println!("Will overwrite the existing output dir!");
            std::fs::remove_dir_all(&outf)?;
        }
    }

    train::train(&args)?;
    Ok(())
}
