"""trailcam/evaluate/metrics.py: accuracy and macro-F1 over the classes present, the per-class
report and the confusion matrix (R17, R32), and the inference timer (R33).

Every expected value is worked out by hand in the test's docstring. The timer runs on one JPEG of
the synthetic fixture (trailcam/data/synthetic.py), written into a temporary directory.
"""

import random
import time
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from trailcam.data import synthetic
from trailcam.data.classes import NUM_CLASSES
from trailcam.data.dataset import EvalTransform
from trailcam.evaluate import metrics
from trailcam.evaluate.metrics import (
    CameraScores,
    ClassScores,
    accuracy,
    confusion_matrix,
    macro_f1,
    per_camera_scores,
    per_class_scores,
    time_one_prediction,
)
from trailcam.model.factory import build_model


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
    for score in (accuracy, macro_f1, per_class_scores, confusion_matrix):
        with pytest.raises(ValueError):
            score(predictions, labels)


def test_per_class_and_confusion_on_the_twelve_class_toy_case() -> None:
    """The 12-class case above: all 12 classes twice, one 11 read as 3.
    Confusion: 2 on the diagonal, but row 11 has 1 at column 11 and 1 at column 3.
    class 3: TP 2, predicted 3, support 2: precision 2/3, recall 1, F1 4/5.
    class 11: TP 1, predicted 1, support 2: precision 1, recall 1/2, F1 2/3. The rest: 1, 1, 1."""
    labels = [c for c in range(NUM_CLASSES) for _ in range(2)]
    predictions = labels[:-1] + [3]
    expected = [[2 if p == y else 0 for p in range(NUM_CLASSES)] for y in range(NUM_CLASSES)]
    expected[11][11] = 1
    expected[11][3] = 1
    assert confusion_matrix(predictions, labels) == expected

    scores = per_class_scores(predictions, labels)
    assert list(scores) == list(range(NUM_CLASSES))
    for c in set(range(NUM_CLASSES)) - {3, 11}:
        assert scores[c] == ClassScores(1.0, 1.0, 1.0, support=2, predicted=2)
    assert scores[3].precision == pytest.approx(2 / 3)
    assert (scores[3].recall, scores[3].support, scores[3].predicted) == (1.0, 2, 3)
    assert scores[3].f1 == pytest.approx(4 / 5)
    assert (scores[11].precision, scores[11].support, scores[11].predicted) == (1.0, 2, 1)
    assert scores[11].recall == pytest.approx(1 / 2)
    assert scores[11].f1 == pytest.approx(2 / 3)


def test_per_class_and_confusion_with_absent_classes() -> None:
    """labels      0 0 0 1 1 2 2 2 2 5 (classes 0, 1, 2 and 5 present; the other eight absent)
    predictions 0 0 1 1 7 2 2 0 9 9 (7 and 9 are absent classes; 5 is never predicted).
    class 0: support 3, predicted 3 (positions 0, 1, 7), TP 2: P 2/3, R 2/3, F1 4/(4+1+1) = 2/3.
    class 1: support 2, predicted 2 (positions 2, 3), TP 1: P 1/2, R 1/2, F1 2/(2+1+1) = 1/2.
    class 2: support 4, predicted 2 (positions 5, 6), TP 2: P 1, R 1/2, F1 4/(4+0+2) = 2/3.
    class 5: support 1, never predicted: P 0 by convention, R 0, F1 0.
    Accuracy 5/10; macro-F1 (2/3 + 1/2 + 2/3 + 0) / 4 = 11/24. The absent 7 and 9 have no
    per-class entry; their predictions show in columns 7 and 9 of the confusion matrix."""
    labels = [0, 0, 0, 1, 1, 2, 2, 2, 2, 5]
    predictions = [0, 0, 1, 1, 7, 2, 2, 0, 9, 9]

    matrix = confusion_matrix(predictions, labels)
    assert len(matrix) == NUM_CLASSES
    assert all(len(row) == NUM_CLASSES for row in matrix)
    nonzero = {(y, p): n for y, row in enumerate(matrix) for p, n in enumerate(row) if n}
    assert nonzero == {
        (0, 0): 2,
        (0, 1): 1,
        (1, 1): 1,
        (1, 7): 1,
        (2, 2): 2,
        (2, 0): 1,
        (2, 9): 1,
        (5, 9): 1,
    }

    scores = per_class_scores(predictions, labels)
    assert list(scores) == [0, 1, 2, 5]
    assert scores[0] == pytest.approx(ClassScores(2 / 3, 2 / 3, 2 / 3, support=3, predicted=3))
    assert scores[1] == pytest.approx(ClassScores(1 / 2, 1 / 2, 1 / 2, support=2, predicted=2))
    assert scores[2] == pytest.approx(ClassScores(1.0, 1 / 2, 2 / 3, support=4, predicted=2))
    assert scores[5] == ClassScores(0.0, 0.0, 0.0, support=1, predicted=0)
    assert accuracy(predictions, labels) == pytest.approx(5 / 10)
    assert macro_f1(predictions, labels) == pytest.approx(11 / 24)


def test_confusion_rows_are_true_and_columns_predicted() -> None:
    """One 4 read as 6: the count sits at row 4, column 6, not row 6, column 4."""
    matrix = confusion_matrix([6], [4])
    assert matrix[4][6] == 1
    assert sum(map(sum, matrix)) == 1


def test_macro_f1_is_the_mean_of_the_per_class_f1() -> None:
    """The identity the module docstring promises, on random cases where some classes are
    absent from the labels and some present classes are never predicted."""
    rng = random.Random(0)
    for _ in range(200):
        present = rng.sample(range(NUM_CLASSES), rng.randint(1, NUM_CLASSES))
        labels = [rng.choice(present) for _ in range(rng.randint(1, 40))]
        predictions = [rng.randrange(NUM_CLASSES) for _ in labels]
        scores = per_class_scores(predictions, labels)
        assert set(scores) == set(labels)
        mean = sum(s.f1 for s in scores.values()) / len(scores)
        assert macro_f1(predictions, labels) == pytest.approx(mean)
        matrix = confusion_matrix(predictions, labels)
        assert sum(map(sum, matrix)) == len(labels)
        assert [sum(matrix[c]) for c in scores] == [s.support for s in scores.values()]


def test_per_camera_scores_by_hand() -> None:
    """Location 7: labels 0, 0, 1, predictions 0, 1, 1: accuracy 2/3; class 0 F1 = 2/(2+0+1) =
    2/3, class 1 F1 = 2/(2+1+0) = 2/3, macro-F1 2/3. Location 3: labels 2, 2, 2, predictions
    2, 2, 5: accuracy 2/3; only class 2 is present there, F1 = 4/(4+0+1) = 0.8, so the
    prediction of the absent class 5 costs recall only. Pooled, macro-F1 would be (2/3 + 2/3 +
    0.8) / 3 = 0.7111 at both: each location is scored on its own images. Keys ascend."""
    predictions = [0, 1, 1, 2, 2, 5]
    labels = [0, 0, 1, 2, 2, 2]
    locations = [7, 7, 7, 3, 3, 3]
    scores = per_camera_scores(predictions, labels, locations)
    assert list(scores) == [3, 7]
    assert scores[7] == CameraScores(3, pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert scores[3] == CameraScores(3, pytest.approx(2 / 3), pytest.approx(0.8))


def test_per_camera_scores_refuse_mismatched_locations() -> None:
    with pytest.raises(ValueError):
        per_camera_scores([0, 1], [0, 1], [5])
    with pytest.raises(ValueError):
        per_camera_scores([0, 12], [0, 1], [5, 5])


def test_the_timer_runs_one_fixture_image_through_the_small_cnn(tmp_path: Path) -> None:
    root = synthetic.write_synthetic_fixture(tmp_path / "fixture").image_root
    path = sorted(root.rglob("*.jpg"))[0]
    with Image.open(path) as image:
        tensor = EvalTransform()(image.convert("RGB"))
    seconds = time_one_prediction(build_model("small_cnn"), tensor)
    assert isinstance(seconds, float)
    assert seconds > 0


class _Recorder(nn.Module):
    """Logs each forward pass: grad mode, train mode, input shape and device."""

    def __init__(self, events: list[str], device: str = "cpu") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, device=device))
        self.events = events
        self.calls: list[tuple[bool, bool, tuple[int, ...], torch.device]] = []

    def forward(self, x: Tensor) -> Tensor:
        self.events.append("forward")
        self.calls.append((torch.is_grad_enabled(), self.training, tuple(x.shape), x.device))
        return x


def test_the_timer_clocks_the_second_forward_pass_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """One warm-up call, then perf_counter read just before and just after the second call, in
    eval mode under no_grad; the clock difference is returned and the train mode restored."""
    events: list[str] = []
    readings = iter([10.0, 10.25])

    def clock() -> float:
        events.append("clock")
        return next(readings)

    monkeypatch.setattr(time, "perf_counter", clock)
    model = _Recorder(events)
    model.train()
    seconds = time_one_prediction(model, torch.zeros(3, 8, 8))
    assert seconds == pytest.approx(0.25)
    assert events == ["forward", "clock", "forward", "clock"]
    assert model.calls == [(False, False, (1, 3, 8, 8), torch.device("cpu"))] * 2
    assert model.training


def test_the_timer_runs_on_the_models_device() -> None:
    """A CPU image is moved to the model's device (here `meta`, which needs no GPU)."""
    model = _Recorder([], device="meta")
    model.eval()
    time_one_prediction(model, torch.zeros(1, 3, 8, 8))
    assert [call[3].type for call in model.calls] == ["meta", "meta"]
    assert not model.training


@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (8, 8), (1, 1, 3, 8, 8)])
def test_the_timer_takes_one_image_only(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        time_one_prediction(_Recorder([]), torch.zeros(shape))


def test_the_timer_is_the_modules_only_clock() -> None:
    """R33: perf_counter, never time.time, in the module that the demo and the Space import."""
    source = Path(metrics.__file__).read_text()
    assert "time.perf_counter()" in source
    assert "time.time(" not in source
