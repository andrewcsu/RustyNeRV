#!/bin/bash
#SBATCH --job-name=nerv-rust-build
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=build_output.log

source "$HOME/.cargo/env"
export LIBTORCH=/nas/longleaf/home/andrewsu/comp590/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

cd /nas/longleaf/home/andrewsu/comp590/nerv-rust
cargo clean -p torch-sys 2>&1
cargo clean -p nerv-rust 2>&1
cargo build --release 2>&1
echo "BUILD_EXIT_CODE=$?"
