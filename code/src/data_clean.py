from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from log import setup_logger


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    class_name: str
    path: Path
    relative_path: Path
    width: int | None
    height: int | None
    mode: str | None
    is_small: bool
    is_valid: bool
    error: str | None = None

    @property
    def short_edge(self) -> int | None:
        if self.width is None or self.height is None:
            return None
        return min(self.width, self.height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview and clean the cats-vs-dogs dataset with an interactive flow."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root directory that contains class subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/cleaned_runs"),
        help="Directory that will contain timestamped cleaned datasets.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("dataset/logs"),
        help="Directory used for preview logs.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("dataset/previews"),
        help="Directory used for preview collage images.",
    )
    parser.add_argument(
        "--small-threshold",
        type=int,
        default=128,
        help="Images with short edge below this value are treated as small.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=6,
        help="How many small images to sample for the preview collage.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260317,
        help="Random seed used when sampling small-image previews.",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def scan_dataset(input_dir: Path, small_threshold: int) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
            if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            relative_path = image_path.relative_to(input_dir)
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                    mode = image.mode

                records.append(
                    ImageRecord(
                        class_name=class_dir.name,
                        path=image_path,
                        relative_path=relative_path,
                        width=width,
                        height=height,
                        mode=mode,
                        is_small=min(width, height) < small_threshold,
                        is_valid=True,
                    )
                )
            except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
                records.append(
                    ImageRecord(
                        class_name=class_dir.name,
                        path=image_path,
                        relative_path=relative_path,
                        width=None,
                        height=None,
                        mode=None,
                        is_small=False,
                        is_valid=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return records


def log_preview(records: list[ImageRecord], logger, small_threshold: int) -> None:
    total_records = len(records)
    valid_records = [record for record in records if record.is_valid]
    invalid_records = [record for record in records if not record.is_valid]
    small_records = [record for record in valid_records if record.is_small]
    non_rgb_records = [record for record in valid_records if record.mode != "RGB"]

    class_counter = Counter(record.class_name for record in records)
    mode_counter = Counter(record.mode for record in valid_records if record.mode is not None)

    width_values = [record.width for record in valid_records if record.width is not None]
    height_values = [record.height for record in valid_records if record.height is not None]

    logger.info("========== Dataset Preview ==========")
    logger.info("Input directory: %s", valid_records[0].path.parents[1] if valid_records else "N/A")
    logger.info("Total scanned files: %s", total_records)
    logger.info("Valid images: %s", len(valid_records))
    logger.info("Invalid images: %s", len(invalid_records))
    logger.info("Small-image rule: short edge < %s", small_threshold)

    logger.info("Class distribution:")
    for class_name, count in sorted(class_counter.items()):
        logger.info("  %s: %s", class_name, count)

    if width_values and height_values:
        logger.info(
            "Image size range: width %s-%s, height %s-%s",
            min(width_values),
            max(width_values),
            min(height_values),
            max(height_values),
        )

    logger.info("Image mode distribution:")
    for mode_name, count in sorted(mode_counter.items(), key=lambda item: (-item[1], item[0])):
        logger.info("  %s: %s", mode_name, count)

    logger.info("Small images found: %s", len(small_records))
    for record in small_records:
        logger.info(
            "  SMALL | %s | %sx%s | mode=%s",
            record.relative_path.as_posix(),
            record.width,
            record.height,
            record.mode,
        )

    logger.info("Non-RGB images to convert: %s", len(non_rgb_records))
    for record in non_rgb_records:
        logger.info("  CONVERT | %s | mode=%s", record.relative_path.as_posix(), record.mode)

    logger.info("Invalid images to exclude: %s", len(invalid_records))
    for record in invalid_records:
        logger.info("  INVALID | %s | %s", record.relative_path.as_posix(), record.error)

    logger.info("=====================================")


def build_collage(
    sampled_records: list[ImageRecord],
    output_path: Path,
    tile_size: tuple[int, int] = (220, 220),
    columns: int = 3,
) -> Path | None:
    if not sampled_records:
        return None

    ensure_directory(output_path.parent)
    rows = (len(sampled_records) + columns - 1) // columns
    label_height = 42
    canvas = Image.new(
        "RGB",
        (columns * tile_size[0], rows * (tile_size[1] + label_height)),
        color=(248, 246, 240),
    )

    for index, record in enumerate(sampled_records):
        row = index // columns
        col = index % columns
        x_offset = col * tile_size[0]
        y_offset = row * (tile_size[1] + label_height)

        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            fitted = ImageOps.contain(image, tile_size)

        image_x = x_offset + (tile_size[0] - fitted.width) // 2
        image_y = y_offset + (tile_size[1] - fitted.height) // 2
        canvas.paste(fitted, (image_x, image_y))

        label = f"{record.class_name}/{record.path.name} {record.width}x{record.height}"
        label_image = Image.new("RGB", (tile_size[0], label_height), color=(230, 226, 217))
        canvas.paste(label_image, (x_offset, y_offset + tile_size[1]))
        draw = ImageDraw.Draw(label_image)
        draw.text((8, 12), label[:32], fill=(30, 30, 30))
        canvas.paste(label_image, (x_offset, y_offset + tile_size[1]))

    canvas.save(output_path, quality=95)
    return output_path


def prompt_for_confirmation() -> bool:
    answer = input("Preview finished. Enter 'y' to continue with cleaning, or any other key to stop: ")
    return answer.strip().lower() == "y"


def prompt_small_image_policy(has_small_images: bool) -> bool:
    if not has_small_images:
        return False

    while True:
        answer = input("Choose small-image policy: [k]eep all small images or [r]emove all small images: ")
        normalized = answer.strip().lower()
        if normalized in {"k", "keep"}:
            return False
        if normalized in {"r", "remove"}:
            return True
        print("Please enter 'k' to keep or 'r' to remove.")


def clean_dataset(
    records: list[ImageRecord],
    output_dir: Path,
    remove_small_images: bool,
    logger,
) -> dict[str, int]:
    ensure_directory(output_dir)
    stats = defaultdict(int)

    for record in records:
        if not record.is_valid:
            stats["excluded_invalid"] += 1
            continue

        if remove_small_images and record.is_small:
            stats["excluded_small"] += 1
            continue

        destination = output_dir / record.relative_path
        ensure_directory(destination.parent)

        if record.mode == "RGB":
            shutil.copy2(record.path, destination)
            stats["copied"] += 1
        else:
            with Image.open(record.path) as image:
                converted = ImageOps.exif_transpose(image).convert("RGB")
                converted.save(destination, format="JPEG", quality=95)
            stats["converted_to_rgb"] += 1

    logger.info("Cleaning complete. Output directory: %s", output_dir)
    logger.info("  copied RGB files: %s", stats["copied"])
    logger.info("  converted to RGB: %s", stats["converted_to_rgb"])
    logger.info("  excluded invalid: %s", stats["excluded_invalid"])
    logger.info("  excluded small: %s", stats["excluded_small"])
    logger.info("  total written: %s", stats["copied"] + stats["converted_to_rgb"])
    return dict(stats)


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = args.log_dir / f"data_preview_{run_id}.log"
    preview_file = args.preview_dir / f"small_image_collage_{run_id}.jpg"
    output_dir = args.output_root / f"cleaned_petimages_{run_id}"

    logger = setup_logger("data_clean", log_file)

    try:
        records = scan_dataset(args.dataset_dir, args.small_threshold)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    log_preview(records, logger, args.small_threshold)

    small_records = [record for record in records if record.is_valid and record.is_small]
    random_generator = random.Random(args.random_seed)
    sample_count = min(args.sample_count, len(small_records))
    sampled_records = random_generator.sample(small_records, sample_count) if sample_count else []
    collage_path = build_collage(sampled_records, preview_file)

    if collage_path is not None:
        logger.info("Small-image collage saved to: %s", collage_path)
        for record in sampled_records:
            logger.info(
                "  SAMPLE | %s | %sx%s",
                record.relative_path.as_posix(),
                record.width,
                record.height,
            )
    else:
        logger.info("No small images were found, so no collage was generated.")

    if not prompt_for_confirmation():
        logger.info("Cleaning cancelled by user before execution.")
        logger.info("Preview log saved to: %s", log_file)
        return 0

    remove_small_images = prompt_small_image_policy(bool(small_records))
    logger.info(
        "User selected small-image policy: %s",
        "remove all small images" if remove_small_images else "keep all small images",
    )

    clean_dataset(
        records=records,
        output_dir=output_dir,
        remove_small_images=remove_small_images,
        logger=logger,
    )
    logger.info("Preview log saved to: %s", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
