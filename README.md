# NeRV-Rust: Neural Representations for Videos (Rust)

A complete Rust rewrite of [NeRV (NeurIPS 2021)](https://arxiv.org/abs/2110.13903) using **tch-rs** (libtorch bindings) with full CUDA/GPU support.

## Prerequisites

- **Rust** (1.70+) via [rustup](https://rustup.rs/)
- **libtorch** (PyTorch C++ library, 2.0+). Set environment variables:
  ```bash
  export LIBTORCH=/path/to/libtorch
  export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
  ```

## Build

```bash
cargo build --release
```

## Data Setup

Place video frames (PNG/JPEG) in `./data/{dataset_name}/`, sorted by filename. For example, for Big Buck Bunny:

```
./data/bunny/
├── frame_0000.png
├── frame_0001.png
├── ...
```

## Training

### NeRV-S on Big Buck Bunny
```bash
cargo run --release -- \
    -e 300 --lower-width 96 --num-blocks 1 --dataset bunny --frame-gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem-dim-num 512_1 --reduction 2 \
    --fc-hw-dim 9_16_26 --expansion 1 --single-res --loss-type Fusion6 \
    --warmup 0.2 --lr-type cosine --strides 5 2 2 2 2 --conv-type conv \
    -b 1 --lr 0.0005 --norm none --act swish
```

For NeRV-M and NeRV-L, change `--fc-hw-dim` to `9_16_58` and `9_16_112` respectively.

## Evaluation

```bash
cargo run --release -- \
    -e 300 --lower-width 96 --num-blocks 1 --dataset bunny --frame-gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem-dim-num 512_1 --reduction 2 \
    --fc-hw-dim 9_16_26 --expansion 1 --single-res --loss-type Fusion6 \
    --warmup 0.2 --lr-type cosine --strides 5 2 2 2 2 --conv-type conv \
    -b 1 --lr 0.0005 --norm none --act swish \
    --weight checkpoints/model.pth --eval-only
```

### Dump Predicted Frames
Add `--dump-images` to the evaluation command.

### FPS Benchmark
Add `--eval-fps` to the evaluation command.

## Model Pruning

### Prune and Fine-tune
```bash
cargo run --release -- \
    -e 100 --lower-width 96 --num-blocks 1 --dataset bunny --frame-gap 1 \
    --outf prune_ab --embed 1.25_40 --stem-dim-num 512_1 --reduction 2 \
    --fc-hw-dim 9_16_26 --expansion 1 --single-res --loss-type Fusion6 \
    --warmup 0.0 --lr-type cosine --strides 5 2 2 2 2 --conv-type conv \
    -b 1 --lr 0.0005 --norm none --act swish \
    --weight checkpoints/model.pth --not-resume-epoch --prune-ratio 0.4
```

### Evaluate Pruned Model with Quantization
```bash
cargo run --release -- \
    -e 100 --lower-width 96 --num-blocks 1 --dataset bunny --frame-gap 1 \
    --outf dbg --embed 1.25_40 --stem-dim-num 512_1 --reduction 2 \
    --fc-hw-dim 9_16_26 --expansion 1 --single-res --loss-type Fusion6 \
    --warmup 0.0 --lr-type cosine --strides 5 2 2 2 2 --conv-type conv \
    -b 1 --lr 0.0005 --norm none --act swish \
    --weight checkpoints/pruned_model.pth --prune-ratio 0.4 \
    --eval-only --quant-bit 8 --quant-axis 0
```

## Bits-per-pixel (bpp)

Final bpp is computed as:

```
bpp = ModelParameters * (1 - ModelSparsity) * QuantBit / PixelNum
```

## Architecture

```
nerv-rust/src/
├── main.rs          # CLI entry point
├── config.rs        # All CLI arguments (clap derive)
├── data.rs          # Frame dataset + mini-batch loader
├── model.rs         # Generator: MLP stem + NeRVBlock conv cascade
├── encoding.rs      # Positional encoding (Fourier features)
├── loss.rs          # L1, L2, SSIM, Fusion1-12 losses
├── ssim.rs          # SSIM and MS-SSIM from scratch
├── metrics.rs       # PSNR and MS-SSIM evaluation metrics
├── scheduler.rs     # LR scheduling (cosine+warmup, step, const)
├── train.rs         # Training loop with multi-scale targets
├── eval.rs          # Evaluation, image dumping, FPS benchmark
├── pruning.rs       # L1 unstructured global pruning
└── quantize.rs      # Weight quantization + Huffman entropy estimation
```

## Citation

```bibtex
@InProceedings{chen2021nerv,
    title={Ne{RV}: Neural Representations for Videos},
    author={Hao Chen and Bo He and Hanyu Wang and Yixuan Ren and Ser-Nam Lim and Abhinav Shrivastava},
    year={2021},
    booktitle={NeurIPS},
}
```
