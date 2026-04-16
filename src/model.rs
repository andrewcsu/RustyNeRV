use tch::nn::{self, Module, ModuleT};
use tch::Tensor;

// ── Activation helpers ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Relu,
    Leaky,
    Leaky01,
    Relu6,
    Gelu,
    Sin,
    Swish,
    Softplus,
    Hardswish,
}

impl Activation {
    pub fn from_str(s: &str) -> Self {
        match s {
            "relu" => Activation::Relu,
            "leaky" => Activation::Leaky,
            "leaky01" => Activation::Leaky01,
            "relu6" => Activation::Relu6,
            "gelu" => Activation::Gelu,
            "sin" => Activation::Sin,
            "swish" => Activation::Swish,
            "softplus" => Activation::Softplus,
            "hardswish" => Activation::Hardswish,
            _ => panic!("Unknown activation: {}", s),
        }
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Relu => x.relu(),
            Activation::Leaky => x.leaky_relu(),
            Activation::Leaky01 => x.where_self(&x.gt(0.0), &(x * 0.1)),
            Activation::Relu6 => x.clamp(0.0, 6.0),
            Activation::Gelu => x.gelu("none"),
            Activation::Sin => x.sin(),
            Activation::Swish => x.silu(),
            Activation::Softplus => x.softplus(),
            Activation::Hardswish => x.hardswish(),
        }
    }
}

// ── Norm helpers ────────────────────────────────────────────────────────

pub enum NormLayer {
    Identity,
    BatchNorm(nn::BatchNorm),
    InstanceNorm,
}

impl NormLayer {
    pub fn new(vs: &nn::Path, norm_type: &str, num_features: i64) -> Self {
        match norm_type {
            "none" => NormLayer::Identity,
            "bn" => {
                let bn = nn::batch_norm2d(vs, num_features, Default::default());
                NormLayer::BatchNorm(bn)
            }
            "in" => { let _ = num_features; NormLayer::InstanceNorm },
            _ => panic!("Unknown norm: {}", norm_type),
        }
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        match self {
            NormLayer::Identity => x.shallow_clone(),
            NormLayer::BatchNorm(bn) => bn.forward_t(x, train),
            NormLayer::InstanceNorm => {
                x.instance_norm(None::<&Tensor>, None::<&Tensor>, None::<&Tensor>, None::<&Tensor>, true, 0.1, 1e-5, false)
            }
        }
    }
}

// ── CustomConv (upsampling module) ──────────────────────────────────────

pub enum CustomConv {
    Conv {
        conv: nn::Conv2D,
        pixel_shuffle_factor: i64,
    },
    Deconv {
        deconv: nn::ConvTranspose2D,
    },
    Bilinear {
        stride: i64,
        conv: nn::Conv2D,
    },
}

impl CustomConv {
    pub fn new(
        vs: &nn::Path,
        ngf: i64,
        new_ngf: i64,
        stride: i64,
        bias: bool,
        conv_type: &str,
    ) -> Self {
        match conv_type {
            "conv" => {
                let out_c = new_ngf * stride * stride;
                let conv = nn::conv2d(
                    vs / "conv",
                    ngf,
                    out_c,
                    3,
                    nn::ConvConfig {
                        padding: 1,
                        bias,
                        ..Default::default()
                    },
                );
                CustomConv::Conv {
                    conv,
                    pixel_shuffle_factor: stride,
                }
            }
            "deconv" => {
                let deconv = nn::conv_transpose2d(
                    vs / "conv",
                    ngf,
                    new_ngf,
                    stride,
                    nn::ConvTransposeConfig {
                        stride,
                        bias,
                        ..Default::default()
                    },
                );
                CustomConv::Deconv { deconv }
            }
            "bilinear" => {
                let kernel = 2 * stride + 1;
                let conv = nn::conv2d(
                    vs / "up_scale",
                    ngf,
                    new_ngf,
                    kernel,
                    nn::ConvConfig {
                        padding: stride,
                        bias,
                        ..Default::default()
                    },
                );
                CustomConv::Bilinear { stride, conv }
            }
            _ => panic!("Unknown conv_type: {}", conv_type),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            CustomConv::Conv {
                conv,
                pixel_shuffle_factor,
            } => {
                let out = conv.forward(x);
                out.pixel_shuffle(*pixel_shuffle_factor)
            }
            CustomConv::Deconv { deconv } => deconv.forward(x),
            CustomConv::Bilinear { stride, conv } => {
                let s = *stride as f64;
                let up = x.upsample_bilinear2d(
                    &[x.size()[2] * *stride, x.size()[3] * *stride],
                    true,
                    Some(s),
                    Some(s),
                );
                conv.forward(&up)
            }
        }
    }
}

// ── NeRVBlock ───────────────────────────────────────────────────────────

pub struct NeRVBlock {
    conv: CustomConv,
    norm: NormLayer,
    act: Activation,
}

impl NeRVBlock {
    pub fn new(
        vs: &nn::Path,
        ngf: i64,
        new_ngf: i64,
        stride: i64,
        bias: bool,
        norm: &str,
        act: &str,
        conv_type: &str,
    ) -> Self {
        let conv = CustomConv::new(&(vs / "conv"), ngf, new_ngf, stride, bias, conv_type);
        let norm_layer = NormLayer::new(&(vs / "norm"), norm, new_ngf);
        let activation = Activation::from_str(act);
        Self {
            conv,
            norm: norm_layer,
            act: activation,
        }
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let out = self.conv.forward(x);
        let out = self.norm.forward_t(&out, train);
        self.act.apply(&out)
    }
}

// ── MLP (stem) ──────────────────────────────────────────────────────────

pub struct Mlp {
    layers: Vec<nn::Linear>,
    act: Activation,
}

impl Mlp {
    pub fn new(vs: &nn::Path, dim_list: &[i64], act: &str, bias: bool) -> Self {
        let activation = Activation::from_str(act);
        let mut layers = Vec::new();
        for i in 0..dim_list.len() - 1 {
            let lin = nn::linear(
                vs / format!("{}", i * 2),
                dim_list[i],
                dim_list[i + 1],
                nn::LinearConfig { bias, ..Default::default() },
            );
            layers.push(lin);
        }
        Self {
            layers,
            act: activation,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.shallow_clone();
        for layer in &self.layers {
            out = layer.forward(&out);
            out = self.act.apply(&out);
        }
        out
    }
}

// ── Generator ───────────────────────────────────────────────────────────

pub struct Generator {
    stem: Mlp,
    fc_h: i64,
    fc_w: i64,
    fc_dim: i64,
    layers: Vec<NeRVBlock>,
    head_layers: Vec<Option<nn::Conv2D>>,
    use_sigmoid: bool,
}

impl Generator {
    pub fn new(
        vs: &nn::Path,
        embed_length: i64,
        stem_dim_num: &str,
        fc_hw_dim: &str,
        expansion: f64,
        num_blocks: i64,
        norm: &str,
        act: &str,
        bias: bool,
        reduction: i64,
        conv_type: &str,
        stride_list: &[i64],
        sin_res: bool,
        lower_width: i64,
        sigmoid: bool,
    ) -> Self {
        let stem_parts: Vec<i64> = stem_dim_num
            .split('_')
            .map(|s| s.parse().unwrap())
            .collect();
        let stem_dim = stem_parts[0];
        let stem_num = stem_parts[1];

        let fc_parts: Vec<i64> = fc_hw_dim
            .split('_')
            .map(|s| s.parse().unwrap())
            .collect();
        let fc_h = fc_parts[0];
        let fc_w = fc_parts[1];
        let fc_dim = fc_parts[2];

        // Build MLP dim list: [embed_length, stem_dim, ..., stem_dim, fc_h*fc_w*fc_dim]
        let mut mlp_dims: Vec<i64> = Vec::new();
        mlp_dims.push(embed_length);
        for _ in 0..stem_num {
            mlp_dims.push(stem_dim);
        }
        mlp_dims.push(fc_h * fc_w * fc_dim);

        let stem = Mlp::new(&(vs / "stem"), &mlp_dims, act, bias);

        let mut layers = Vec::new();
        let mut head_layers: Vec<Option<nn::Conv2D>> = Vec::new();
        let mut ngf = fc_dim;

        for (i, &stride) in stride_list.iter().enumerate() {
            let new_ngf = if i == 0 {
                (ngf as f64 * expansion) as i64
            } else {
                let divisor = if stride == 1 { 1 } else { reduction };
                (ngf / divisor).max(lower_width)
            };

            for j in 0..num_blocks {
                let effective_stride = if j == 0 { stride } else { 1 };
                let block = NeRVBlock::new(
                    &(vs / "layers" / format!("{}", layers.len())),
                    ngf,
                    new_ngf,
                    effective_stride,
                    bias,
                    norm,
                    act,
                    conv_type,
                );
                layers.push(block);
                ngf = new_ngf;

                // Head layers are appended per-block to match zip(layers, head_layers) in forward
                if j < num_blocks - 1 {
                    head_layers.push(None);
                }
            }

            // After all blocks for this stride stage, add the head
            let head = if sin_res {
                if i == stride_list.len() - 1 {
                    Some(nn::conv2d(
                        vs / "head_layers" / format!("{}", i),
                        ngf,
                        3,
                        1,
                        nn::ConvConfig { bias, ..Default::default() },
                    ))
                } else {
                    None
                }
            } else {
                Some(nn::conv2d(
                    vs / "head_layers" / format!("{}", i),
                    ngf,
                    3,
                    1,
                    nn::ConvConfig { bias, ..Default::default() },
                ))
            };
            head_layers.push(head);
        }

        Self {
            stem,
            fc_h,
            fc_w,
            fc_dim,
            layers,
            head_layers,
            use_sigmoid: sigmoid,
        }
    }

    /// Forward pass. Returns a list of RGB image tensors (one per active head).
    pub fn forward_t(&self, input: &Tensor, train: bool) -> Vec<Tensor> {
        let mut output = self.stem.forward(input);
        let bs = output.size()[0];
        output = output.view([bs, self.fc_dim, self.fc_h, self.fc_w]);

        let mut out_list = Vec::new();
        for (layer, head) in self.layers.iter().zip(self.head_layers.iter()) {
            output = layer.forward_t(&output, train);
            if let Some(head_conv) = head {
                let img_out = head_conv.forward(&output);
                let img_out = if self.use_sigmoid {
                    img_out.sigmoid()
                } else {
                    (img_out.tanh() + 1.0) * 0.5
                };
                out_list.push(img_out);
            }
        }
        out_list
    }
}

/// Count total trainable parameters in a VarStore.
pub fn count_parameters(vs: &nn::VarStore) -> f64 {
    vs.variables()
        .into_iter()
        .map(|(_, t)| t.numel() as f64)
        .sum::<f64>()
        / 1e6
}
