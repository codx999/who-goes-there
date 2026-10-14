"""Scores from predicted and true class indices (plan.md section 7; R17, R32).

Accuracy and macro-F1 are reported side by side everywhere a score is (R32). Macro-F1 averages
the per-class F1 over the classes present in the true labels, so a split that lacks some classes
(trans_val has 9 of the 12) is not pulled down by classes it cannot contain. A prediction of an
absent class still costs: it is a miss for the true class, whose recall drops. A present class
that is never predicted has precision 0 by convention, so its F1 is 0.

The per-class report and the confusion matrix keep the same conventions, so `macro_f1` is the mean
of `per_class_scores` F1 over the classes it returns. The confusion matrix is indexed by the class
order of `trailcam/data/classes.py`: rows are the true class, columns the predicted class.

No scikit-learn: the counts are plain Python over index sequences, checked against hand-computed
values in `tests/test_metrics.py`. Item 5.03 wrote `accuracy` and `macro_f1` for the training
loop; item 8.01 added the per-class report, the confusion matrix and the inference timer.

`time_one_prediction` (R33) is the one timer: the Gradio UI and the Space call it rather than
time a prediction their own way. It times the model's forward pass only. Decoding the image and
the eval transform happen before it is called and are not in the timed region, nor is moving the
input to the model's device; the seconds the demo shows are the model's.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

from trailcam.data.classes import NUM_CLASSES


def _check(predictions: Sequence[int], labels: Sequence[int]) -> None:
    if len(predictions) != len(labels):
        raise ValueError(f"{len(predictions)} predictions for {len(labels)} labels")
    if not labels:
        raise ValueError("no labels to score")
    for name, values in (("prediction", predictions), ("label", labels)):
        bad = sum(1 for v in values if not 0 <= v < NUM_CLASSES)
        if bad:
            raise ValueError(f"{bad} {name}s outside the class indices 0 to {NUM_CLASSES - 1}")


def accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """The share of predictions equal to their label."""
    _check(predictions, labels)
    return sum(p == y for p, y in zip(predictions, labels, strict=True)) / len(labels)


def confusion_matrix(predictions: Sequence[int], labels: Sequence[int]) -> list[list[int]]:
    """`NUM_CLASSES` x `NUM_CLASSES` counts: row = true class, column = predicted class, both in
    the class order of `trailcam/data/classes.py`. A class absent from `labels` has a row of
    zeros."""
    _check(predictions, labels)
    matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for p, y in zip(predictions, labels, strict=True):
        matrix[y][p] += 1
    return matrix


class ClassScores(NamedTuple):
    """One class's scores. `support` is its count in the labels, `predicted` how often it was
    predicted."""

    precision: float
    recall: float
    f1: float
    support: int
    predicted: int


def per_class_scores(predictions: Sequence[int], labels: Sequence[int]) -> dict[int, ClassScores]:
    """Precision, recall and F1 for each class present in `labels`, keyed by class index in
    ascending order; absent classes have no entry, as in `macro_f1`. A present class that is never
    predicted has precision 0. F1 = 2TP / (2TP + FP + FN), which is 0 when TP is 0."""
    matrix = confusion_matrix(predictions, labels)
    scores = {}
    for c in sorted(set(labels)):
        tp = matrix[c][c]
        support = sum(matrix[c])
        predicted = sum(row[c] for row in matrix)
        fp = predicted - tp
        fn = support - tp
        scores[c] = ClassScores(
            precision=tp / predicted if predicted else 0.0,
            recall=tp / support,
            f1=2 * tp / (2 * tp + fp + fn),
            support=support,
            predicted=predicted,
        )
    return scores


def macro_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """The mean F1 over the classes present in `labels` (R32), each F1 = 2TP / (2TP + FP + FN)."""
    scores = per_class_scores(predictions, labels)
    return sum(s.f1 for s in scores.values()) / len(scores)


def _synchronize(device: torch.device) -> None:
    # GPU kernels run asynchronously; wait for them so the clock stops after the forward pass.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def time_one_prediction(model: nn.Module, image: Tensor) -> float:
    """Seconds for one forward pass of `model` on one preprocessed image (R33).

    `image` is the eval transform's output, (3, H, W), or a batch of one, (1, 3, H, W). It is
    moved to the model's device first; then one untimed warm-up call, then `time.perf_counter()`
    around a second forward pass, in eval mode under `torch.no_grad()`. On a GPU the clock stops
    after the device has finished. The model's train or eval mode is restored afterwards.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"expected one image, (3, H, W) or (1, 3, H, W), got {tuple(image.shape)}")
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    batch = image.to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(batch)  # warm-up, not timed
            _synchronize(device)
            start = time.perf_counter()
            model(batch)
            _synchronize(device)
            return time.perf_counter() - start
    finally:
        model.train(was_training)
