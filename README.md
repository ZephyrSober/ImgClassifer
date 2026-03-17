# ImgClassifer
This is a image classifer based on MindSpore. It basicly identify [cats] and [dogs]. More labels will be supported in the future.

# Dataset
We use a part of a larger dataset on kaggle. For origin dataset visit https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

# Data Preview And Cleaning
The interactive preprocessing entry is:

```bash
python code/src/interactive_data_cleaning.py --dataset-dir dataset/kagglecatsanddogs_3367a/PetImages
```

Optional arguments:

```bash
python code/src/interactive_data_cleaning.py \
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

# Logging
Reusable logging utilities are placed under `code/src/log/` so the same logger setup can be reused later in training scripts.
