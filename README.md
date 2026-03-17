# ImgClassifer
This is a image classifer based on MindSpore. It basicly identify [cats] and [dogs]. More labels will be supported in the future.

# Dataset
We use a part of a larger dataset on kaggle. For origin dataset visit https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

# Data Preview And Cleaning
The interactive preprocessing entry is:

```bash
python code/src/data_clean.py --dataset-dir dataset/kagglecatsanddogs_3367a/PetImages
```

Optional arguments:

```bash
python code/src/data_clean.py \
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
python code/src/data_split.py
```

By default it uses the latest directory under `dataset/cleaned_runs`, keeps a fixed test seed, and writes a manifest-driven split under `dataset/splits/<cleaned_run>/`.

Example:

```bash
python code/src/data_split.py \
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

# Logging
Reusable logging utilities are placed under `code/src/log/` so the same logger setup can be reused later in training scripts.

# Developer Notes
- Generalization design for scaling the dataset pipeline beyond cats-vs-dogs:
  [code/src/dataset_pipeline_generalization_design.md](/d:/1FILE/Program/python/ImgClassifer/code/src/dataset_pipeline_generalization_design.md)
