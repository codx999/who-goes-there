"""Scores from predicted and true class indices (plan.md section 7; R17, R32).

Accuracy and macro-F1 are reported side by side everywhere a score is (R32). Macro-F1 averages
the per-class F1 over the classes present in the true labels, so a split that lacks some classes
(trans_val has 9 of the 12) is not pulled down by classes it cannot contain. A prediction of an
absent class still costs: it is a miss for the true class, whose recall drops. A present class
that is never predicted has precision 0 by convention, so its F1 is 0.

No scikit-learn: the counts are plain Python over index sequences, checked against hand-computed
values in `tests/test_metrics.py`. Item 5.03 wrote these two functions for the training loop;
item 8.01 adds the per-class report, the confusion matrix and the inference timer here.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def macro_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """The mean F1 over the classes present in `labels` (R32), each F1 = 2TP / (2TP + FP + FN)."""
    _check(predictions, labels)
    tp = [0] * NUM_CLASSES
    fp = [0] * NUM_CLASSES
    fn = [0] * NUM_CLASSES
    for p, y in zip(predictions, labels, strict=True):
        if p == y:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1
    present = sorted(set(labels))
    return sum(2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) for c in present) / len(present)
