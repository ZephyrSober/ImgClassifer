# ImgClassifer
This is a image classifer based on MindSpore. It basicly identify [cats] and [dogs]. More labels will be supported in the future.

# Dataset
We use a part of a larger dataset on kaggle. For origin dataset visit https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

# Data Preview And Cleaning
The interactive preprocessing entry is:

```bash
python code/data_src/data_clean.py --dataset-dir dataset/kagglecatsanddogs_3367a/PetImages
```

Optional arguments:

```bash
python code/data_src/data_clean.py \
  --dataset-dir dataset/kagglecatsanddogs_3367a/PetImages \
  --output-root dataset/cleaned_runs \
  --log-dir dataset/logs \
  --preview-dir dataset/previews \
  --small-threshold 128 \
  --sample-count 6 \
  --random-seed 20260317
```

What the script does:

1. Scan the dataset directory passed by `--dataset-dir`.
2. Print the preview summary to the console.
3. Save the same preview content to a log file.
4. Sample small images and generate a collage preview.
5. Wait for manual confirmation. Cleaning only continues after input `y`.
6. Ask whether to keep all small images or remove all small images.
7. Write the cleaned dataset to a timestamped directory under `dataset/cleaned_runs`.

Current cleaning rules:

- Invalid images are excluded.
- Non-RGB images are converted to RGB.
- Small images are defined as images with short edge `< 128`.
- Small-image retention is decided interactively during execution.

# Dataset Splitting
The split generation entry is:

```bash
python code/data_src/data_split.py
```

By default it uses the latest directory under `dataset/cleaned_runs`, keeps a fixed test seed, and writes a manifest-driven split under `dataset/splits/<cleaned_run>/`.

Example:

```bash
python code/data_src/data_split.py \
  --cleaned-run-dir dataset/cleaned_runs/cleaned_petimages_20260317_164602_652096 \
  --output-root dataset/splits \
  --log-dir dataset/logs \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --test-seed 20260317 \
  --val-seed 20260318
```

What the script does:

1. Read a cleaned dataset run and infer the `run_id`.
2. Recover converted-image metadata from the matching preview log when available.
3. Rescan image sizes to build difficulty-aware strata using class, small-image flag, size bucket, aspect bucket, and conversion flag.
4. Generate a fixed `test` split with `--test-seed`.
5. Generate a reproducible `val` split from the remaining samples with `--val-seed`.
6. Save `manifest.csv` and `summary.json` under a versioned output directory.
7. Run integrity checks so the same sample or group cannot leak across splits.

Manifest fields include:

- `run_id`, `cleaned_run`, `relative_path`, `label`, `split`
- `group_id`, `width`, `height`, `short_edge`, `long_edge`
- `aspect_ratio`, `aspect_bucket`, `size_bucket`
- `is_small`, `is_extreme_aspect`, `difficulty_tag`
- `current_mode`, `original_mode`, `was_converted_to_rgb`
- `test_seed`, `val_seed`

Optional group-aware splitting:

- Pass `--group-manifest path/to/groups.csv`
- The CSV must contain `relative_path,group_id`
- Samples with the same `group_id` will stay in the same split
- This is useful later if you add near-duplicate detection or source-based grouping

# Dataloader Usage
The training-side dataloader is implemented under `code/train/src/data/` and reads samples from the generated `manifest.csv`.

Current public interfaces:

- `build_dataloader(dataset_cfg, train_cfg, runtime_cfg, split)`
- `build_dataloaders(dataset_cfg, train_cfg, runtime_cfg)`

Before using it, make sure these prerequisites are satisfied:

1. Install `mindspore`, `numpy`, and `Pillow` in your training environment.
2. Run dataset cleaning and split generation first.
3. Make sure `dataset_root` points to a real cleaned dataset directory and `manifest_path` points to a real split manifest.

The default dataset config already points to:

- cleaned images: `dataset/cleaned_runs/cleaned_petimages_20260317_164602_652096`
- manifest: `dataset/splits/cleaned_petimages_20260317_164602_652096/split_test20260317_val20260318/manifest.csv`

If those files are not present on your machine, update `code/train/configs/dataset/cats_dogs_manifest.yaml` before building the dataloader.

Minimal example:

```python
from pathlib import Path
import sys

import yaml

repo_root = Path(".").resolve()
sys.path.insert(0, str(repo_root / "code" / "train" / "src"))

from data import build_dataloaders


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

dataset_cfg = load_yaml(repo_root / "code/train/configs/dataset/cats_dogs_manifest.yaml")
train_cfg = load_yaml(repo_root / "code/train/configs/train/sgd_baseline.yaml")
runtime_cfg = load_yaml(repo_root / "code/train/configs/runtime/cpu.yaml")

dataloaders, num_classes = build_dataloaders(dataset_cfg, train_cfg, runtime_cfg)
train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]

print("num_classes =", num_classes)
print("train dataset size =", train_loader.get_dataset_size())

for batch in train_loader.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(batch["image"].shape)  # [batch, 3, H, W]
    print(batch["label"].shape)  # [batch]
    break
```

If you only want one split:

```python
from data import build_dataloader

train_loader = build_dataloader(dataset_cfg, train_cfg, runtime_cfg, "train")
```

What the dataloader does:

- reads only from `manifest.csv` instead of inferring splits from folder names
- validates required columns, label names, split values, and image paths during initialization
- loads images as RGB
- applies randomized transforms for `train`
- applies deterministic transforms for `val` and `test`
- batches with `drop_remainder=True` for `train` and `False` for `val/test`

Current transform behavior:

- `train`: resize shortest edge, random crop, random horizontal flip, normalize, HWC to CHW
- `val/test`: resize shortest edge, center crop, normalize, HWC to CHW

If `mindspore` is not installed, importing the public dataloader builder will raise an `ImportError`.

# Logging
Reusable logging utilities are placed under `code/data_src/log/` so the same logger setup can be reused later in training scripts.

# Developer Notes
- Generalization design for scaling the dataset pipeline beyond cats-vs-dogs:
  `./code/data_src/dataset_pipeline_generalization_design.md`
