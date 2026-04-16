use anyhow::Result;
use std::path::{Path, PathBuf};
use tch::Tensor;

pub struct CustomDataSet {
    main_dir: PathBuf,
    frame_paths: Vec<String>,
    frame_indices: Vec<f32>,
    frame_gap: usize,
}

impl CustomDataSet {
    pub fn new(
        main_dir: &str,
        vid_list: Option<&[usize]>,
        frame_gap: usize,
    ) -> Result<Self> {
        let dir = Path::new(main_dir);
        let mut all_imgs: Vec<String> = std::fs::read_dir(dir)?
            .filter_map(|e| {
                let e = e.ok()?;
                let name = e.file_name().into_string().ok()?;
                if e.path().is_file() {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();
        all_imgs.sort();

        let total = all_imgs.len();
        let mut frame_paths = Vec::with_capacity(total);
        let mut frame_indices = Vec::with_capacity(total);

        for (i, img_name) in all_imgs.into_iter().enumerate() {
            frame_paths.push(img_name);
            frame_indices.push(i as f32 / total as f32);
        }

        if let Some(vl) = vid_list {
            let filtered_paths: Vec<String> = vl.iter().map(|&i| frame_paths[i].clone()).collect();
            let filtered_idx: Vec<f32> = vl.iter().map(|&i| frame_indices[i]).collect();
            Ok(Self {
                main_dir: dir.to_path_buf(),
                frame_paths: filtered_paths,
                frame_indices: filtered_idx,
                frame_gap,
            })
        } else {
            Ok(Self {
                main_dir: dir.to_path_buf(),
                frame_paths,
                frame_indices,
                frame_gap,
            })
        }
    }

    pub fn len(&self) -> usize {
        self.frame_indices.len() / self.frame_gap
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Load a single sample: returns (image_tensor [C,H,W], frame_index scalar tensor).
    pub fn get(&self, idx: usize) -> Result<(Tensor, Tensor)> {
        let valid_idx = idx * self.frame_gap;
        let img_path = self.main_dir.join(&self.frame_paths[valid_idx]);
        let img = image::open(&img_path)?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as i64, rgb.height() as i64);

        let raw: Vec<f32> = rgb.as_raw().iter().map(|&b| b as f32 / 255.0).collect();
        // image crate gives HWC layout; reshape to CHW
        let tensor = Tensor::from_slice(&raw)
            .reshape(&[h, w, 3])
            .permute(&[2, 0, 1]); // -> [C, H, W]

        // Rotate landscape: if height > width, swap H and W (transpose spatial dims)
        let tensor = if tensor.size()[1] > tensor.size()[2] {
            tensor.permute(&[0, 2, 1])
        } else {
            tensor
        };

        let frame_idx = Tensor::from_slice(&[self.frame_indices[valid_idx]]);
        Ok((tensor, frame_idx.squeeze()))
    }

    /// Build a full batch of all items, returns (images [N,C,H,W], indices [N]).
    #[allow(dead_code)]
    pub fn load_all(&self) -> Result<(Tensor, Tensor)> {
        let n = self.len();
        let mut images = Vec::with_capacity(n);
        let mut indices = Vec::with_capacity(n);
        for i in 0..n {
            let (img, idx) = self.get(i)?;
            images.push(img.unsqueeze(0));
            indices.push(idx.unsqueeze(0));
        }
        let images = Tensor::cat(&images, 0);
        let indices = Tensor::cat(&indices, 0);
        Ok((images, indices))
    }
}

/// Simple mini-batch iterator over a dataset.
pub struct DataLoader {
    dataset: CustomDataSet,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
}

impl DataLoader {
    pub fn new(dataset: CustomDataSet, batch_size: usize, shuffle: bool) -> Self {
        let n = dataset.len();
        let indices: Vec<usize> = (0..n).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    pub fn len(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }

    #[allow(dead_code)]
    pub fn dataset_len(&self) -> usize {
        self.dataset.len()
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.indices.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.pos..end];
        self.pos = end;

        let mut images = Vec::with_capacity(batch_indices.len());
        let mut frame_indices = Vec::with_capacity(batch_indices.len());
        for &i in batch_indices {
            let (img, idx) = self.dataset.get(i).expect("failed to load image");
            images.push(img.unsqueeze(0));
            frame_indices.push(idx.unsqueeze(0));
        }
        let images = Tensor::cat(&images, 0);
        let frame_indices = Tensor::cat(&frame_indices, 0);
        Some((images, frame_indices))
    }
}
