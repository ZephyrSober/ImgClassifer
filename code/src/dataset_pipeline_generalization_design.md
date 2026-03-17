# Dataset Pipeline Generalization Design

## Summary
This document describes how to generalize the current dataset preprocessing pipeline so it can scale from the current cats-vs-dogs project to larger image-classification datasets such as ImageNet.

The current implementation already works for directory-based classification datasets, but it is still tightly coupled to:

- the cats-vs-dogs cleaning workflow
- timestamped `cleaned_runs` directories
- preview-log parsing for metadata recovery
- full-image rescans in multiple stages
- interactive behavior that does not scale well to very large datasets

The refactor proposed here is a design only. It does not change current scripts yet.

## Current State

### Existing strengths
- Both [data_clean.py](/d:/1FILE/Program/python/ImgClassifer/code/src/data_clean.py) and [data_split.py](/d:/1FILE/Program/python/ImgClassifer/code/src/data_split.py) already support directory-per-class datasets.
- The current split flow is manifest-driven at the output stage.
- The split logic already supports reproducible seeds and optional group-aware splitting.
- The cleaning logic already extracts useful metadata such as image size, mode, and small-image flags.

### Current bottlenecks
- Cleaning and splitting both rescan image files directly.
- Metadata is not treated as a first-class artifact.
- `data_split.py` depends on parsing cleaning logs to recover `original_mode`.
- Naming assumptions such as `cleaned_petimages_<run_id>` are project-specific.
- Per-image logging is too verbose for large datasets.
- The pipeline assumes a custom split must always be generated, which does not fit datasets with official train/val/test definitions.

## Refactor Goals
- Make the pipeline dataset-agnostic for directory-based image classification datasets.
- Make metadata the central source of truth for cleaning, splitting, and later training.
- Remove project-specific assumptions from storage layout and naming.
- Support both custom splits and externally defined official splits.
- Keep reproducibility guarantees.
- Make the pipeline performant enough for much larger datasets.
- Preserve backward compatibility for the current cats-vs-dogs project during migration.

## Target Architecture

### Core principle
The pipeline should become `manifest-first`.

Instead of having each stage independently rescan the filesystem and infer state, the pipeline should produce explicit structured metadata artifacts and let downstream stages consume them.

### Proposed stages
1. `scan`
   Read dataset files and produce a raw inventory manifest.
2. `clean`
   Apply cleaning policy using the inventory manifest and emit a cleaned manifest plus optional cleaned files.
3. `group`
   Optionally derive duplicate/source groups for leakage prevention.
4. `split`
   Generate split assignments from a manifest, not from raw directories.
5. `train-consume`
   Training code reads one manifest and filters rows by `split`.

### Proposed modules
- `dataset_schema.py`
  Shared dataclasses or typed row schemas.
- `dataset_scan.py`
  Fast dataset inventory builder.
- `dataset_clean.py`
  Manifest-driven cleaning implementation.
- `dataset_group.py`
  Optional group generation from hashes, metadata, or external rules.
- `dataset_split.py`
  Manifest-driven split generator.
- `dataset_io.py`
  Shared CSV/JSONL/parquet read-write helpers.
- `dataset_stats.py`
  Summary and validation utilities.

The current [data_clean.py](/d:/1FILE/Program/python/ImgClassifer/code/src/data_clean.py) and [data_split.py](/d:/1FILE/Program/python/ImgClassifer/code/src/data_split.py) should eventually become thin CLI wrappers over these shared modules.

## Canonical Manifest Design

### Why this matters
The most important technical change is introducing a canonical manifest schema that survives across stages.

### Recommended manifest levels

#### 1. Inventory manifest
Produced immediately after scanning the original dataset.

Recommended columns:
- `dataset_id`
- `source_root`
- `relative_path`
- `label`
- `source_split`
- `width`
- `height`
- `short_edge`
- `long_edge`
- `aspect_ratio`
- `mode`
- `suffix`
- `file_size_bytes`
- `is_valid`
- `validation_error`
- `sha256` or `phash` if enabled
- `group_id` if externally known

#### 2. Cleaned manifest
Produced after cleaning decisions are applied.

Recommended columns:
- all inventory columns that remain relevant
- `clean_run_id`
- `clean_status`
- `exclude_reason`
- `output_relative_path`
- `output_mode`
- `was_converted_to_rgb`
- `is_small`
- `size_bucket`
- `aspect_bucket`
- `difficulty_tag`

#### 3. Split manifest
Produced after split assignment.

Recommended columns:
- all cleaned-manifest columns needed by training
- `split_run_id`
- `split`
- `split_strategy`
- `test_seed`
- `val_seed`
- `stratum_key`

### File format recommendation
- Start with CSV for simplicity and inspectability.
- Add optional JSONL or parquet support later for very large datasets.
- Keep column names stable across pipeline stages.

## Dataset Abstraction

### Current assumption
The current code assumes:
- directory names under root are labels
- every file under a class directory is a sample
- there is no predefined split

### Proposed abstraction
Define a dataset descriptor with these concepts:
- `dataset_type`
  Example: `directory_classification`, `imagenet_style`, `manifest_defined`
- `root_dir`
- `label_source`
  Example: directory name, annotation file, external CSV
- `split_source`
  Example: none, official directory, official manifest, external override
- `group_source`
  Example: none, supplied manifest, derived hashes

This allows the same pipeline to support:
- current cats-vs-dogs layout
- ImageNet-style class folders
- datasets with official validation sets
- datasets where labels live in annotation files instead of directory names

## Cleaning Refactor

### Problems with current implementation
- Cleaning logic mixes scanning, preview, decision-making, file writing, and logging in one script.
- Metadata is discovered during cleaning but not persisted in a reusable structured artifact.
- Interactive prompts are built into the primary workflow.

### Target cleaning behavior
Cleaning should accept an input manifest and a cleaning policy, then emit:
- a cleaned manifest
- a cleaning summary
- optional rewritten cleaned files

### Policy model
Replace prompt-specific logic with a policy object such as:
- `exclude_invalid`
- `convert_non_rgb`
- `small_image_policy`
  values: `keep`, `exclude`, `tag_only`
- `small_threshold`
- `min_width`
- `min_height`
- `allowed_suffixes`

### Important design choice
For large datasets, `tag_only` should become the default for many quality signals.

Example:
- keep small images in the dataset
- record `is_small = true`
- let split/training/evaluation decide how to treat them

This is more flexible than physically deleting them early.

## Split Refactor

### Problems with current implementation
- Split generation rescans cleaned images instead of consuming a cleaned manifest.
- It recovers `original_mode` from preview logs, which is fragile.
- It assumes every dataset needs a custom split.
- Fine-grained strata may become too sparse for datasets with many classes.

### Target split behavior
Split generation should accept:
- a cleaned manifest
- a split strategy config
- optional group manifest overrides

Then emit:
- a split manifest
- a split summary
- validation reports

### Split strategy types
Support multiple strategies explicitly:
- `official`
  Respect predefined dataset split assignments.
- `stratified_random`
  Random split with stable seeds and metadata-aware strata.
- `group_stratified`
  Same as above, but group-aware.
- `train_official_val_custom_test`
  Example hybrid mode when only part of the split is predefined.
- `cross_validation`
  For later expansion.

### Sparse-strata handling
For large multi-class datasets, the split logic needs a fallback hierarchy.

Recommended fallback order:
1. `label + size_bucket + aspect_bucket + conversion_flag`
2. `label + size_bucket + conversion_flag`
3. `label`

If a stratum is too small to support all target splits, the code should automatically back off to a coarser key and record that decision in the summary.

### Group-aware support
The current optional `group_manifest` idea should stay, but it should become a standard input path rather than an add-on.

Group IDs may come from:
- external metadata
- filename/source identifiers
- content hashes
- near-duplicate detection

## Logging And Summaries

### Current issue
Per-sample logs are useful for small datasets, but not for large-scale runs.

### Proposed logging approach
Use two output levels:

#### 1. Structured artifacts
- manifest files
- summary JSON
- validation report JSON

#### 2. Human-readable logs
- aggregate counts
- top-N examples only
- warnings for anomalies

For large runs, avoid writing every small image or every converted file into the log stream.

## Performance Considerations

### Current performance risks
- repeated PIL open operations
- full rescans in separate stages
- single-threaded execution
- CSV-only workflows becoming slow at very large scale

### Proposed improvements
- scan once, reuse manifests downstream
- optionally add multiprocessing for image metadata extraction
- make hashing optional because it is expensive
- support incremental runs by caching prior scan results
- separate metadata generation from file rewriting

### Recommended ordering
For large datasets:
1. scan metadata only
2. review summary
3. choose cleaning policy
4. materialize cleaned files only if necessary

This prevents expensive rewrites before policy is stable.

## Storage Layout

### Current layout
- `dataset/cleaned_runs/<timestamp>`
- `dataset/splits/<cleaned_run>/<seed combo>`

### Proposed generalized layout
```text
dataset/
  sources/
    <dataset_id>/
  manifests/
    <dataset_id>/
      inventory/
      cleaned/
      groups/
      splits/
  materialized/
    <dataset_id>/
      cleaned/
```

Benefits:
- metadata and materialized files are separated
- multiple split strategies can point at the same cleaned manifest
- datasets are identified by stable dataset IDs rather than project-specific names

## Backward Compatibility

### Requirement
The cats-vs-dogs workflow should continue to work during migration.

### Migration plan
1. Introduce shared manifest schema and IO helpers.
2. Make `data_clean.py` write a canonical cleaned manifest in addition to current outputs.
3. Make `data_split.py` prefer cleaned manifests and only fall back to rescanning for backward compatibility.
4. Add dataset IDs and generalized naming while still accepting legacy `cleaned_petimages_*` directories.
5. Later split the current monolithic scripts into reusable modules.

This staged migration reduces risk and keeps current commands working.

## Training Integration

### Target behavior
Training should consume the split manifest directly.

The dataloader should:
- read one manifest
- filter by `split`
- map `relative_path` or `output_relative_path` to image files
- use manifest metadata for analysis and curriculum options

### Why this matters
This removes the need to physically copy separate `train/val/test` directory trees for every experiment.

## ImageNet-Specific Notes

### What already transfers well
- directory-based labels
- image validation and mode normalization
- manifest-based split outputs
- group-aware split extension points

### What does not transfer cleanly yet
- current naming assumptions
- interactive cleaning flow
- preview-log dependency
- current stratum design without fallback
- lack of explicit support for official splits

### Recommended mode for ImageNet-like datasets
- use scan + manifest generation first
- preserve official splits when they exist
- treat quality rules mostly as tags, not exclusions
- use group-aware splitting only when generating custom subsets or derived tasks

## Validation Requirements

The generalized pipeline should validate:
- every manifest row has a unique sample key
- split assignments cover all retained rows exactly once
- no group leaks across splits
- label distributions are within tolerance
- sparse-strata fallback events are recorded
- referenced image files exist when materialized output is expected

## Non-Goals
- supporting object detection or segmentation annotations in this refactor
- replacing the training pipeline in the same step
- introducing a database-backed metadata store
- implementing distributed preprocessing in the first iteration

## Recommended First Refactor Slice
If this design is implemented later, the safest first slice is:

1. Add a canonical cleaned manifest writer to the current cleaning pipeline.
2. Refactor split generation to consume that manifest instead of rescanning files.
3. Remove preview-log parsing by storing `original_mode` directly in the cleaned manifest.
4. Add split-strategy fallback for sparse strata.

This single slice gives most of the portability benefits without requiring a full rewrite.

## Decision Summary
- Make metadata artifacts the primary interface between stages.
- Generalize dataset identity and layout.
- Decouple cleaning policy from interactive prompts.
- Support official and custom split strategies explicitly.
- Add fallback logic for sparse strata.
- Keep current CLI behavior temporarily via compatibility wrappers.
