#!/bin/bash
#SBATCH --job-name=nerv-eval
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=eval_output.log

source "$HOME/.cargo/env"
module load cuda/11.8
export LIBTORCH=/nas/longleaf/home/andrewsu/comp590/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

cd /nas/longleaf/home/andrewsu/comp590/nerv-rust

CKPT="output/bunny_ab/bunny/embed1.25_40_512_1_fc_9_16_26__exp1_reduce2_low96_blk1_cycle1_gap1_e300_warm60_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_actswish_/model_val_best.pth"

./target/release/nerv-rust \
    -e 300 --lower-width 96 --num-blocks 1 --dataset bunny --frame-gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem-dim-num 512_1 --reduction 2 \
    --fc-hw-dim 9_16_26 --expansion 1 --single-res --loss-type Fusion6 \
    --warmup 0.2 --lr-type cosine --strides 5 2 2 2 2 --conv-type conv \
    -b 1 --lr 0.0005 --norm none --act swish \
    --weight "$CKPT" --eval-only --dump-images
