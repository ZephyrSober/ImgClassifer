from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from log import setup_logger


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
LOG_RUN_ID_PATTERN = re.compile(r"data_preview_(?P<run_id>\d{8}_\d{6}(?:_\d+)?)\.log$")
CONVERT_PATTERN = re.compile(r"CONVERT \| (?P<path>.+?) \| mode=(?P<mode>.+)$")


@dataclass(frozen=True)
class SplitRecord:
    run_id: str
    cleaned_run: str
    relative_path: str
    label: str
    split: str
    group_id: str
    width: int
    height: int
    short_edge: int
    long_edge: int
    aspect_ratio: float
    aspect_bucket: str
    size_bucket: str
    is_small: bool
    is_extreme_aspect: bool
    difficulty_tag: str
    current_mode: str
    original_mode: str
    was_converted_to_rgb: bool
    test_seed: int
    val_seed: int


@dataclass(frozen=True)
class ImageSample:
    relative_path: Path
    class_name: str
    path: Path
    width: int
    height: int
    current_mode: str
    original_mode: str
    was_converted_to_rgb: bool
    size_bucket: str
    aspect_bucket: str
    is_small: bool
    is_extreme_aspect: bool
    difficulty_tag: str
    group_id: str

    @property
    def short_edge(self) -> int:
        return min(self.width, self.height)

    @property
    def long_edge(self) -> int:
        return max(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        return round(self.width / self.height, 6)

    @property
    def stratum_key(self) -> str:
        return "|".join(
            [
                self.class_name,
                f"small={int(self.is_small)}",
                f"converted={int(self.was_converted_to_rgb)}",
                f"size={self.size_bucket}",
                f"aspect={self.aspect_bucket}",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test manifests for a cleaned cats-vs-dogs dataset run."
    )
    parser.add_argument(
        "--cleaned-run-dir",
        type=Path,
        default=None,
        help="Path to a cleaned dataset under dataset/cleaned_runs. Defaults to the latest cleaned run.",
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path("dataset/cleaned_runs"),
        help="Directory that stores timestamped cleaned datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/splits"),
        help="Directory that stores versioned split manifests.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("dataset/logs"),
        help="Directory used for split-generation logs and cleaned-run preview logs.",
    )
    parser.add_argument(
        "--small-threshold",
        type=int,
        default=128,
        help="Images with short edge below this value are considered small.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Target train ratio. Must sum with val/test to 1.0.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Target validation ratio. Must sum with train/test to 1.0.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Target test ratio. Must sum with train/val to 1.0.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=20260317,
        help="Stable seed for test-set allocation. Keep this fixed across experiments.",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=20260318,
        help="Seed for validation allocation inside the non-test remainder.",
    )
    parser.add_argument(
        "--group-manifest",
        type=Path,
        default=None,
        help="Optional CSV with columns relative_path,group_id for group-aware splitting.",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    for name, value in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)):
        if not 0 < value < 1:
            raise ValueError(f"{name} ratio must be between 0 and 1, got {value}")


def resolve_cleaned_run_dir(cleaned_root: Path, cleaned_run_dir: Path | None) -> Path:
    if cleaned_run_dir is not None:
        if not cleaned_run_dir.exists():
            raise FileNotFoundError(f"Cleaned run directory does not exist: {cleaned_run_dir}")
        return cleaned_run_dir

    candidates = sorted(
        (path for path in cleaned_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No cleaned runs found under: {cleaned_root}")
    return candidates[-1]


def infer_run_id(cleaned_run_dir: Path) -> str:
    prefix = "cleaned_petimages_"
    name = cleaned_run_dir.name
    return name[len(prefix) :] if name.startswith(prefix) else name


def find_preview_log(log_dir: Path, run_id: str) -> Path | None:
    expected = log_dir / f"data_preview_{run_id}.log"
    if expected.exists():
        return expected

    for path in sorted(log_dir.glob("data_preview_*.log")):
        match = LOG_RUN_ID_PATTERN.match(path.name)
        if match and match.group("run_id") == run_id:
            return path
    return None


def parse_original_modes(preview_log: Path | None) -> dict[str, str]:
    if preview_log is None or not preview_log.exists():
        return {}

    original_modes: dict[str, str] = {}
    with preview_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = CONVERT_PATTERN.search(line)
            if not match:
                continue
            relative_path = match.group("path").replace("\\", "/")
            original_modes[relative_path] = match.group("mode")
    return original_modes


def load_group_mapping(group_manifest: Path | None) -> dict[str, str]:
    if group_manifest is None:
        return {}
    if not group_manifest.exists():
        raise FileNotFoundError(f"Group manifest does not exist: {group_manifest}")

    mapping: dict[str, str] = {}
    with group_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "group_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Group manifest must contain columns {sorted(required)}, got {reader.fieldnames}"
            )
        for row in reader:
            relative_path = str(row["relative_path"]).replace("\\", "/").strip()
            group_id = str(row["group_id"]).strip()
            if not relative_path or not group_id:
                raise ValueError("Group manifest rows must contain non-empty relative_path and group_id")
            mapping[relative_path] = group_id
    return mapping


def build_size_bucket(short_edge: int) -> str:
    if short_edge < 128:
        return "tiny"
    if short_edge < 224:
        return "small"
    if short_edge < 384:
        return "medium"
    return "large"


def build_aspect_bucket(width: int, height: int) -> tuple[str, bool]:
    ratio = width / height
    if ratio >= 1.75:
        return "very_wide", True
    if ratio >= 1.35:
        return "wide", False
    if ratio <= 0.57:
        return "very_tall", True
    if ratio <= 0.74:
        return "tall", False
    return "balanced", False


def build_difficulty_tag(is_small: bool, is_extreme_aspect: bool, was_converted_to_rgb: bool) -> str:
    active = []
    if is_small:
        active.append("small")
    if is_extreme_aspect:
        active.append("extreme_aspect")
    if was_converted_to_rgb:
        active.append("converted")
    return "+".join(active) if active else "standard"


def scan_cleaned_dataset(
    cleaned_run_dir: Path,
    small_threshold: int,
    original_modes: dict[str, str],
    group_mapping: dict[str, str],
) -> list[ImageSample]:
    samples: list[ImageSample] = []
    for class_dir in sorted(path for path in cleaned_run_dir.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
            if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    current_mode = image.mode
            except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
                raise ValueError(f"Unable to read cleaned image {image_path}: {exc}") from exc

            relative_path = image_path.relative_to(cleaned_run_dir)
            relative_key = relative_path.as_posix()
            original_mode = original_modes.get(relative_key, current_mode)
            short_edge = min(width, height)
            is_small = short_edge < small_threshold
            size_bucket = build_size_bucket(short_edge)
            aspect_bucket, is_extreme_aspect = build_aspect_bucket(width, height)
            was_converted_to_rgb = original_mode != current_mode
            difficulty_tag = build_difficulty_tag(is_small, is_extreme_aspect, was_converted_to_rgb)
            group_id = group_mapping.get(relative_key, relative_key)

            samples.append(
                ImageSample(
                    relative_path=relative_path,
                    class_name=class_dir.name,
                    path=image_path,
                    width=width,
                    height=height,
                    current_mode=current_mode,
                    original_mode=original_mode,
                    was_converted_to_rgb=was_converted_to_rgb,
                    size_bucket=size_bucket,
                    aspect_bucket=aspect_bucket,
                    is_small=is_small,
                    is_extreme_aspect=is_extreme_aspect,
                    difficulty_tag=difficulty_tag,
                    group_id=group_id,
                )
            )
    if not samples:
        raise ValueError(f"No supported images found under: {cleaned_run_dir}")
    return samples


def stable_group_order_key(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def allocate_by_ratio(samples: list[ImageSample], ratio: float, seed: int) -> tuple[list[ImageSample], list[ImageSample]]:
    grouped: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_id].append(sample)

    ordered_group_ids = sorted(grouped, key=lambda group_id: stable_group_order_key(group_id, seed))
    target = round(len(samples) * ratio)
    selected: list[ImageSample] = []
    remaining: list[ImageSample] = []
    selected_count = 0

    for group_id in ordered_group_ids:
        members = grouped[group_id]
        if selected_count < target:
            selected.extend(members)
            selected_count += len(members)
        else:
            remaining.extend(members)
    return selected, remaining


def validate_group_consistency(samples: list[ImageSample]) -> None:
    grouped: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_id].append(sample)

    for group_id, members in grouped.items():
        labels = {sample.class_name for sample in members}
        strata = {sample.stratum_key for sample in members}
        if len(labels) > 1:
            raise ValueError(
                f"Group {group_id!r} contains multiple labels {sorted(labels)}; group-aware splitting requires label-consistent groups."
            )
        if len(strata) > 1:
            raise ValueError(
                f"Group {group_id!r} spans multiple strata; regroup samples so a group stays within one stratum."
            )


def assign_splits(
    samples: list[ImageSample],
    test_ratio: float,
    val_ratio: float,
    test_seed: int,
    val_seed: int,
) -> dict[str, str]:
    validate_group_consistency(samples)
    strata: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        strata[sample.stratum_key].append(sample)

    assignments: dict[str, str] = {}
    for stratum_samples in strata.values():
        test_samples, remainder = allocate_by_ratio(stratum_samples, test_ratio, test_seed)
        adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
        val_samples, train_samples = allocate_by_ratio(remainder, adjusted_val_ratio, val_seed)

        for sample in test_samples:
            assignments[sample.relative_path.as_posix()] = "test"
        for sample in val_samples:
            assignments[sample.relative_path.as_posix()] = "val"
        for sample in train_samples:
            assignments[sample.relative_path.as_posix()] = "train"
    return assignments


def build_manifest_rows(
    samples: list[ImageSample],
    assignments: dict[str, str],
    cleaned_run_dir: Path,
    test_seed: int,
    val_seed: int,
) -> list[SplitRecord]:
    run_id = infer_run_id(cleaned_run_dir)
    cleaned_run = cleaned_run_dir.name
    rows: list[SplitRecord] = []

    for sample in sorted(samples, key=lambda item: item.relative_path.as_posix()):
        relative_key = sample.relative_path.as_posix()
        split = assignments[relative_key]
        rows.append(
            SplitRecord(
                run_id=run_id,
                cleaned_run=cleaned_run,
                relative_path=relative_key,
                label=sample.class_name,
                split=split,
                group_id=sample.group_id,
                width=sample.width,
                height=sample.height,
                short_edge=sample.short_edge,
                long_edge=sample.long_edge,
                aspect_ratio=sample.aspect_ratio,
                aspect_bucket=sample.aspect_bucket,
                size_bucket=sample.size_bucket,
                is_small=sample.is_small,
                is_extreme_aspect=sample.is_extreme_aspect,
                difficulty_tag=sample.difficulty_tag,
                current_mode=sample.current_mode,
                original_mode=sample.original_mode,
                was_converted_to_rgb=sample.was_converted_to_rgb,
                test_seed=test_seed,
                val_seed=val_seed,
            )
        )
    return rows


def summarize_manifest(rows: list[SplitRecord], cleaned_run_dir: Path) -> dict[str, object]:
    split_names = ("train", "val", "test")
    split_counts = Counter(row.split for row in rows)
    label_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    small_counts = Counter()
    extreme_aspect_counts = Counter()
    converted_counts = Counter()
    size_bucket_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    aspect_bucket_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        label_counts[row.split][row.label] += 1
        size_bucket_counts[row.split][row.size_bucket] += 1
        aspect_bucket_counts[row.split][row.aspect_bucket] += 1
        if row.is_small:
            small_counts[row.split] += 1
        if row.is_extreme_aspect:
            extreme_aspect_counts[row.split] += 1
        if row.was_converted_to_rgb:
            converted_counts[row.split] += 1

    for split_name in split_names:
        split_counts.setdefault(split_name, 0)
        small_counts.setdefault(split_name, 0)
        extreme_aspect_counts.setdefault(split_name, 0)
        converted_counts.setdefault(split_name, 0)
        label_counts.setdefault(split_name, defaultdict(int))
        size_bucket_counts.setdefault(split_name, defaultdict(int))
        aspect_bucket_counts.setdefault(split_name, defaultdict(int))

    ratios = {split: round(split_counts[split] / len(rows), 6) for split in split_names}

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cleaned_run_dir": str(cleaned_run_dir),
        "cleaned_run": cleaned_run_dir.name,
        "run_id": infer_run_id(cleaned_run_dir),
        "total_samples": len(rows),
        "split_counts": {split: split_counts[split] for split in split_names},
        "split_ratios": ratios,
        "label_counts": {split: dict(sorted(label_counts[split].items())) for split in split_names},
        "small_image_counts": {split: small_counts[split] for split in split_names},
        "extreme_aspect_counts": {split: extreme_aspect_counts[split] for split in split_names},
        "converted_to_rgb_counts": {split: converted_counts[split] for split in split_names},
        "size_bucket_counts": {
            split: dict(sorted(size_bucket_counts[split].items()))
            for split in split_names
        },
        "aspect_bucket_counts": {
            split: dict(sorted(aspect_bucket_counts[split].items()))
            for split in split_names
        },
    }
    return summary


def run_integrity_checks(rows: list[SplitRecord]) -> list[str]:
    issues: list[str] = []
    seen_paths: dict[str, str] = {}

    for row in rows:
        previous = seen_paths.get(row.relative_path)
        if previous is not None and previous != row.split:
            issues.append(f"Path {row.relative_path} appears in multiple splits: {previous}, {row.split}")
        seen_paths[row.relative_path] = row.split

    split_by_path_count = len(seen_paths)
    if split_by_path_count != len(rows):
        issues.append(
            f"Expected unique paths in manifest, got {len(rows)} rows but only {split_by_path_count} unique relative paths."
        )

    if not {"train", "val", "test"}.issubset({row.split for row in rows}):
        issues.append("Manifest does not contain all train, val, and test splits.")

    grouped_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped_splits[row.group_id].add(row.split)
    leaking_groups = [group_id for group_id, splits in grouped_splits.items() if len(splits) > 1]
    if leaking_groups:
        issues.append(f"Found {len(leaking_groups)} groups assigned to multiple splits.")

    return issues


def write_manifest(rows: list[SplitRecord], manifest_path: Path) -> None:
    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(summary: dict[str, object], summary_path: Path) -> None:
    ensure_directory(summary_path.parent)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    cleaned_run_dir = resolve_cleaned_run_dir(args.cleaned_root, args.cleaned_run_dir)
    run_id = infer_run_id(cleaned_run_dir)
    split_name = f"split_test{args.test_seed}_val{args.val_seed}"
    run_output_dir = args.output_root / cleaned_run_dir.name / split_name
    log_file = args.log_dir / f"data_split_{run_id}_{args.test_seed}_{args.val_seed}.log"
    logger = setup_logger("data_split", log_file)

    preview_log = find_preview_log(args.log_dir, run_id)
    original_modes = parse_original_modes(preview_log)
    group_mapping = load_group_mapping(args.group_manifest)

    logger.info("Using cleaned run: %s", cleaned_run_dir)
    logger.info("Split output directory: %s", run_output_dir)
    logger.info("Preview log used for original modes: %s", preview_log if preview_log else "not found")
    logger.info("Group manifest: %s", args.group_manifest if args.group_manifest else "not provided")
    logger.info(
        "Ratios train/val/test: %.3f / %.3f / %.3f",
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    logger.info("Seeds test=%s val=%s", args.test_seed, args.val_seed)

    samples = scan_cleaned_dataset(
        cleaned_run_dir=cleaned_run_dir,
        small_threshold=args.small_threshold,
        original_modes=original_modes,
        group_mapping=group_mapping,
    )
    assignments = assign_splits(
        samples=samples,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        test_seed=args.test_seed,
        val_seed=args.val_seed,
    )
    rows = build_manifest_rows(
        samples=samples,
        assignments=assignments,
        cleaned_run_dir=cleaned_run_dir,
        test_seed=args.test_seed,
        val_seed=args.val_seed,
    )

    issues = run_integrity_checks(rows)
    if issues:
        for issue in issues:
            logger.error(issue)
        raise ValueError("Split integrity checks failed.")

    summary = summarize_manifest(rows, cleaned_run_dir)
    manifest_path = run_output_dir / "manifest.csv"
    summary_path = run_output_dir / "summary.json"
    write_manifest(rows, manifest_path)
    write_summary(summary, summary_path)

    logger.info("Manifest saved to: %s", manifest_path)
    logger.info("Summary saved to: %s", summary_path)
    logger.info("Split counts: %s", summary["split_counts"])
    logger.info("Small image counts: %s", summary["small_image_counts"])
    logger.info("Converted-to-RGB counts: %s", summary["converted_to_rgb_counts"])
    logger.info("Split generation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
