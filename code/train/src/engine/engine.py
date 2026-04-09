from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mindspore as ms
from mindspore import nn

from metrics import compute_accuracy


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def train_one_epoch(
    model,
    dataset,
    loss_fn,
    optimizer,
    epoch: int,
    total_epochs: int,
    log_interval: int = 10,
    logger: logging.Logger | None = None,
) -> float:
    model.set_train(True)
    if logger is not None:
        logger.info("Preparing train step cell for epoch %s/%s", epoch, total_epochs)
    build_start = time.perf_counter()
    train_step = nn.TrainOneStepCell(nn.WithLossCell(model, loss_fn), optimizer)
    train_step.set_train()
    if logger is not None:
        logger.info(
            "Train step cell prepared for epoch %s/%s in %.2fs",
            epoch,
            total_epochs,
            time.perf_counter() - build_start,
        )
    total_loss = 0.0
    steps = dataset.get_dataset_size()
    if steps <= 0:
        raise ValueError("Training dataset is empty.")

    effective_interval = max(1, int(log_interval))
    if logger is not None:
        logger.info("Epoch %s/%s started with %s training steps", epoch, total_epochs, steps)
    epoch_start = time.perf_counter()
    for batch_idx, (data, label) in enumerate(dataset.create_tuple_iterator(), start=1):
        batch_fetch_done = time.perf_counter()
        if batch_idx == 1 and logger is not None:
            logger.info(
                "First batch fetched for epoch %s/%s after %.2fs",
                epoch,
                total_epochs,
                batch_fetch_done - epoch_start,
            )
            logger.info("First train step started for epoch %s/%s", epoch, total_epochs)
        step_start = time.perf_counter()
        loss = train_step(data, label)
        if batch_idx == 1 and logger is not None:
            logger.info(
                "First train step finished for epoch %s/%s in %.2fs",
                epoch,
                total_epochs,
                time.perf_counter() - step_start,
            )
        loss_value = float(loss.asnumpy())
        total_loss += loss_value

        if batch_idx % effective_interval == 0 or batch_idx == steps:
            message = (
                f"Epoch [{epoch}/{total_epochs}] Step [{batch_idx}/{steps}] "
                f"Loss: {loss_value:.4f}"
            )
            if logger is None:
                print(message)
            else:
                logger.info(message)

    if logger is not None:
        logger.info(
            "Epoch %s/%s training finished in %.2fs",
            epoch,
            total_epochs,
            time.perf_counter() - epoch_start,
        )
    return total_loss / steps


def validate_one_epoch(model, dataset, loss_fn) -> tuple[float, float]:
    model.set_train(False)
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    steps = dataset.get_dataset_size()
    if steps <= 0:
        raise ValueError("Validation dataset is empty.")

    val_start = time.perf_counter()
    for data, label in dataset.create_tuple_iterator():
        logits = model(data)
        loss = loss_fn(logits, label)
        total_loss += float(loss.asnumpy())

        batch_correct, batch_total = compute_accuracy(logits, label)
        correct_preds += batch_correct
        total_samples += batch_total

    avg_loss = total_loss / steps
    accuracy = correct_preds / total_samples
    _ = val_start
    return avg_loss, accuracy


def test_one_epoch(model, dataset, loss_fn) -> tuple[float, float]:
    return validate_one_epoch(model, dataset, loss_fn)


def save_checkpoint_if_best(
    model,
    current_acc: float,
    best_acc: float,
    save_dir: str | Path = "./checkpoints",
    model_name: str = "best.ckpt",
    epoch: int | None = None,
) -> tuple[float, Path | None]:
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    if current_acc > best_acc:
        if epoch is None:
            save_path = save_dir_path / model_name
        else:
            save_path = save_dir_path / f"best_epoch_{epoch:03d}.ckpt"
        ms.save_checkpoint(model, str(save_path))
        return current_acc, save_path
    return best_acc, None


def _save_last_checkpoint(model, checkpoints_dir: Path, epoch: int) -> Path:
    save_path = checkpoints_dir / f"last_epoch_{epoch:03d}.ckpt"
    ms.save_checkpoint(model, str(save_path))
    return save_path


def _append_epoch_metrics(metrics_path: Path, row: dict[str, Any]) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "val_accuracy"]
    write_header = not metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_final_metrics(metrics_path: Path, payload: dict[str, Any]) -> None:
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def run_training(
    model,
    dataloaders: dict[str, Any],
    loss_fn,
    optimizer,
    train_cfg: Any,
    run_paths: Mapping[str, Path],
    logger: logging.Logger,
) -> dict[str, Any]:
    epochs = int(_read_config(train_cfg, "epochs", 0))
    log_interval = int(_read_config(train_cfg, "log_interval", 10))
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    checkpoints_dir = Path(run_paths["checkpoints_dir"])
    epoch_metrics_path = Path(run_paths["epoch_metrics_path"])
    final_metrics_path = Path(run_paths["final_metrics_path"])

    best_acc = -1.0
    best_epoch = -1
    best_ckpt_path = checkpoints_dir / "best.ckpt"
    last_ckpt_path = checkpoints_dir / "last.ckpt"

    for epoch in range(1, epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, epochs)
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            dataset=dataloaders["train"],
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=epochs,
            log_interval=log_interval,
            logger=logger,
        )
        logger.info("Starting validation for epoch %s/%s", epoch, epochs)
        validation_start = time.perf_counter()
        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            dataset=dataloaders["val"],
            loss_fn=loss_fn,
        )
        logger.info(
            "Validation for epoch %s/%s finished in %.2fs",
            epoch,
            epochs,
            time.perf_counter() - validation_start,
        )
        logger.info(
            "Epoch %s/%s finished: train_loss=%.4f, val_loss=%.4f, val_accuracy=%.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_accuracy,
        )

        logger.info("Saving last checkpoint for epoch %s", epoch)
        last_ckpt_path = _save_last_checkpoint(model, checkpoints_dir, epoch)
        updated_best_acc, maybe_best_path = save_checkpoint_if_best(
            model=model,
            current_acc=val_accuracy,
            best_acc=best_acc,
            save_dir=checkpoints_dir,
            model_name="best.ckpt",
            epoch=epoch,
        )
        if updated_best_acc > best_acc:
            best_acc = updated_best_acc
            best_epoch = epoch
            best_ckpt_path = maybe_best_path or best_ckpt_path
            logger.info("New best checkpoint saved at epoch %s: %s", best_epoch, best_ckpt_path)

        _append_epoch_metrics(
            epoch_metrics_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{val_accuracy:.6f}",
            },
        )
        logger.info(
            "Epoch %s/%s total elapsed time: %.2fs",
            epoch,
            epochs,
            time.perf_counter() - epoch_start,
        )

    if best_ckpt_path.exists():
        logger.info("Loading best checkpoint before final test: %s", best_ckpt_path)
        params = ms.load_checkpoint(str(best_ckpt_path))
        ms.load_param_into_net(model, params)

    logger.info("Starting final test evaluation")
    test_start = time.perf_counter()
    test_loss, test_accuracy = test_one_epoch(
        model=model,
        dataset=dataloaders["test"],
        loss_fn=loss_fn,
    )
    logger.info(
        "Final test finished: test_loss=%.4f, test_accuracy=%.4f",
        test_loss,
        test_accuracy,
    )
    logger.info("Final test evaluation finished in %.2fs", time.perf_counter() - test_start)

    result = {
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
        "best_ckpt_path": str(best_ckpt_path),
        "last_ckpt_path": str(last_ckpt_path),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    _write_final_metrics(final_metrics_path, result)
    return result
