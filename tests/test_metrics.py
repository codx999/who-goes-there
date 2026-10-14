"""trailcam/evaluate/metrics.py: accuracy and macro-F1 over the classes present (R17, R32).

Every expected value is worked out by hand in the test's docstring. Item 8.01 adds the per-class
report, the confusion matrix and the timer to the module and their tests here.
"""

import pytest

from trailcam.data.classes import NUM_CLASSES
from trailcam.evaluate.metrics import accuracy, macro_f1


def test_accuracy_is_the_share_of_correct_predictions() -> None:
    """3 of 5 right."""
    assert accuracy([0, 1, 1, 1, 0], [0, 0, 1, 1, 2]) == pytest.approx(0.6)


def test_macro_f1_by_hand() -> None:
    """labels 0 0 1 1 2, predictions 0 1 1 1 0.
    class 0: TP 1, FP 1 (the 2 read as 0), FN 1 (a 0 read as 1): F1 2/(2+1+1) = 0.5.
    class 1: TP 2, FP 1, FN 0: F1 4/5 = 0.8. class 2: TP 0, FN 1: F1 0.
    Mean over the three present classes: 1.3 / 3."""
    assert macro_f1([0, 1, 1, 1, 0], [0, 0, 1, 1, 2]) == pytest.approx(1.3 / 3)


def test_macro_f1_averages_over_the_classes_present_only() -> None:
    """labels 0 0 1, predictions 0 5 1: class 0 F1 2/(2+0+1) = 2/3, class 1 F1 1. Class 5 is
    absent from the labels, so it is not averaged: (2/3 + 1) / 2 = 5/6, not (5/3) / 12. The
    wrong 5 still costs class 0 its recall."""
    assert macro_f1([0, 5, 1], [0, 0, 1]) == pytest.approx(5 / 6)


def test_a_twelve_class_toy_case_with_absent_classes() -> None:
    """All 12 classes, two images each, perfect but for one 11 read as 3:
    class 3: TP 2, FP 1: F1 4/5; class 11: TP 1, FN 1: F1 2/3; the other ten 1.
    Mean (10 + 0.8 + 2/3) / 12. Then the same predictions restricted to the nine classes 0 to 8
    (trans_val has 9 of 12) average over nine."""
    labels = [c for c in range(NUM_CLASSES) for _ in range(2)]
    predictions = labels[:-1] + [3]
    assert accuracy(predictions, labels) == pytest.approx(23 / 24)
    assert macro_f1(predictions, labels) == pytest.approx((10 + 0.8 + 2 / 3) / 12)
    nine = [c for c in range(9) for _ in range(2)]
    nine_predictions = nine[:-1] + [11]  # an 8 read as the absent 11
    # class 8: TP 1, FN 1: F1 2/3; classes 0 to 7: 1. Mean (8 + 2/3) / 9.
    assert macro_f1(nine_predictions, nine) == pytest.approx((8 + 2 / 3) / 9)


def test_a_class_never_predicted_scores_zero() -> None:
    """Everything read as 0: class 0 TP 1, FP 2: F1 2/4; classes 1 and 2: F1 0. Mean 1/6."""
    assert macro_f1([0, 0, 0], [0, 1, 2]) == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    ("predictions", "labels"),
    [([0], [0, 1]), ([], []), ([12], [0]), ([0], [-1])],
)
def test_bad_input_is_refused(predictions: list[int], labels: list[int]) -> None:
    for score in (accuracy, macro_f1):
        with pytest.raises(ValueError):
            score(predictions, labels)
