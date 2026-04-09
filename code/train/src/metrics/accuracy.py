from __future__ import annotations

import mindspore as ms
from mindspore import ops


def compute_accuracy(logits, labels) -> tuple[int, int]:
    preds = ops.Argmax(axis=1)(logits)
    matches = ops.Cast()(preds == labels, ms.float32)
    correct = int(ops.ReduceSum()(matches).asnumpy())
    total = int(labels.shape[0])
    return correct, total
