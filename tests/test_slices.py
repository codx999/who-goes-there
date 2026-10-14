"""trailcam/evaluate/slices.py: day/night and common/rare slices and the feature map (plan.md
section 7; R17 test isolation and the rare rule fixed before scoring, R32 accuracy beside
macro-F1 over the classes present, R03 counts and scores only, R04 figures only under --out).

Slice scores are checked against hand-computed values and must add up to the split; the PCA
against a case with known components; the end-to-end tests run on the synthetic fixture with
random weights, no download.
"""

import dataclasses
import re
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image

from trailcam.data import synthetic
from trailcam.data.classes import CLASSES, index_of_name
from trailcam.data.dataset import CCT20Dataset, EvalTransform
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Record, load_metadata
from trailcam.data.split import Split, make_split
from trailcam.evaluate import slices
from trailcam.evaluate.metrics import accuracy, macro_f1
from trailcam.evaluate.slices import (
    CAMERA_FIGURE,
    RARE_CLASSES,
    RARE_SHARE,
    SPECIES_FIGURE,
    TABLE_NAME,
    SplitPredictions,
    mix_rows,
    pca_2d,
    predict_split,
    project,
    rare_classes,
    render_report,
    slice_rows,
    write_feature_maps,
)
from trailcam.model.factory import build_model, head_module
from trailcam.model.small_cnn import SmallCNN
from trailcam.train.train import CACHE_ANNOTATIONS, CACHE_IMAGES, file_sha256

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
OPOSSUM, RABBIT, BIRD, SKUNK = (index_of_name(n) for n in ("opossum", "rabbit", "bird", "skunk"))


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Split, Path]:
    tmp = tmp_path_factory.mktemp("slices")
    written = synthetic.write_synthetic_fixture(tmp)
    split = make_split(load_metadata(tmp, written.paths["sidecar"]), SEED)
    return split, written.image_root


def _run(
    split: str,
    labels: list[int],
    predictions: list[int],
    night: list[bool],
    locations: list[int] | None = None,
    features: torch.Tensor | None = None,
) -> SplitPredictions:
    return SplitPredictions(
        split=split,
        labels=tuple(labels),
        predictions=tuple(predictions),
        night=tuple(night),
        locations=tuple(locations or [1] * len(labels)),
        features=features,
    )


# The common / rare rule


def test_the_rare_classes_follow_the_rule_on_the_data_card_train_counts() -> None:
    """Below 4% of the seed-42 train images: bird, rodent and skunk; car, the next class, is
    above the line, so the rule is not one image away from a different cut."""
    card = (ROOT / "docs" / "data-card.md").read_text(encoding="utf-8")
    counts = {}
    for name in CLASSES:
        match = re.search(rf"^\| {name} \| (\d+) \|", card, flags=re.MULTILINE)
        assert match, name
        counts[index_of_name(name)] = int(match.group(1))
    assert sum(counts.values()) == 13587
    assert rare_classes(counts) == RARE_CLASSES
    assert {CLASSES[c] for c in RARE_CLASSES} == {"bird", "rodent", "skunk"}
    total = sum(counts.values())
    common = [n / total for c, n in counts.items() if c not in RARE_CLASSES]
    assert min(common) > RARE_SHARE + 0.003
    assert max(counts[c] / total for c in RARE_CLASSES) < RARE_SHARE - 0.002


def test_rare_classes_refuses_an_empty_count() -> None:
    with pytest.raises(ValueError, match="no train images"):
        rare_classes({})


# Slice tables


def test_the_slices_add_up_to_the_split_and_score_by_hand() -> None:
    #           day      day    night  night  night  day
    labels = [OPOSSUM, RABBIT, BIRD, OPOSSUM, SKUNK, BIRD]
    predicted = [OPOSSUM, OPOSSUM, BIRD, RABBIT, OPOSSUM, SKUNK]
    night = [False, False, True, True, True, False]
    run = _run("val", labels, predicted, night)
    rows = {(r.cut, r.slice): r for r in slice_rows(run)}
    for cut, (first, second) in slices.CUTS.items():
        whole = rows[cut, "all"]
        assert whole.images == len(labels)
        assert rows[cut, first].images + rows[cut, second].images == whole.images
        positions = set(run.in_slice(first)) | set(run.in_slice(second))
        assert positions == set(range(len(labels)))
        assert not set(run.in_slice(first)) & set(run.in_slice(second))
        assert whole.accuracy == accuracy(predicted, labels)
        assert whole.macro_f1 == macro_f1(predicted, labels)
    assert all(r.group == "cis" and r.split == "val" for r in rows.values())

    day, nights = rows["day/night", "day"], rows["day/night", "night"]
    assert (day.images, nights.images) == (3, 3)
    assert day.accuracy == pytest.approx(1 / 3)
    assert nights.accuracy == pytest.approx(1 / 3)
    # Day: opossum TP1 FP1 FN0 -> 2/3; rabbit 0; bird 0 (never predicted right). Mean of 3.
    assert day.classes == 3
    assert day.macro_f1 == pytest.approx((2 / 3) / 3)
    # Night: bird 1, opossum 0, skunk 0, over three present classes.
    assert nights.macro_f1 == pytest.approx(1 / 3)

    common, rare = rows["species", "common"], rows["species", "rare"]
    assert (common.images, rare.images) == (3, 3)
    assert common.accuracy == pytest.approx(1 / 3)
    assert rare.accuracy == pytest.approx(1 / 3)
    assert (common.classes, rare.classes) == (2, 2)
    # Rare: bird TP1 FP0 FN1 -> 2/3, skunk TP0 FP1 FN1 -> 0. Opossum predictions are misses only.
    assert rare.macro_f1 == pytest.approx((2 / 3) / 2)


def test_an_empty_slice_has_no_score_and_the_report_says_so() -> None:
    run = _run("trans_val", [OPOSSUM, RABBIT], [OPOSSUM, OPOSSUM], [True, True])
    rows = {(r.cut, r.slice): r for r in slice_rows(run)}
    assert rows["day/night", "day"].images == 0
    assert rows["day/night", "day"].accuracy is None
    assert rows["day/night", "day"].macro_f1 is None
    assert rows["species", "rare"].images == 0
    table = render_report({"arm A": [run]})
    assert "n/a" in table
    assert "### Cis and trans at the same day/night mix" not in table


def test_the_trans_accuracy_is_reweighted_to_the_cis_night_share() -> None:
    # cis: 4 images, 1 at night (p = 0.25), all right. trans: 2 day (1 right), 2 night (0 right).
    cis = _run("val", [OPOSSUM] * 4, [OPOSSUM] * 4, [True, False, False, False])
    trans = _run(
        "trans_val",
        [OPOSSUM] * 4,
        [OPOSSUM, RABBIT, RABBIT, RABBIT],
        [False, False, True, True],
    )
    (row,) = mix_rows([trans, cis])
    assert (row.cis_split, row.trans_split) == ("val", "trans_val")
    assert row.cis_accuracy == 1.0
    assert row.trans_accuracy == 0.25
    assert row.cis_night_share == 0.25
    assert row.trans_night_share == 0.5
    assert row.trans_at_cis_mix == pytest.approx(0.75 * 0.5 + 0.25 * 0.0)
    assert row.gap == 0.75
    assert row.gap_at_cis_mix == pytest.approx(1.0 - 0.375)
    table = render_report({"arm A": [cis, trans]})
    assert (
        "| val | trans_val | 1.0000 | 0.2500 | 0.7500 | 0.250 | 0.500 | 0.3750 | 0.6250 |" in table
    )

    night_only = _run("trans_val", [OPOSSUM], [OPOSSUM], [True])
    (row,) = mix_rows([cis, night_only])
    assert row.trans_at_cis_mix is None and row.gap_at_cis_mix is None
    assert mix_rows([cis]) == ()
    assert mix_rows([cis, _run("trans_test", [OPOSSUM], [OPOSSUM], [True])]) == ()


def test_the_report_puts_accuracy_beside_macro_f1_and_states_the_rule() -> None:
    run = _run("val", [OPOSSUM, BIRD], [OPOSSUM, BIRD], [False, True])
    table = render_report({"resnet18 arm A": [run], "resnet18 arm B": [run]}, ["a header"])
    assert "a header" in table
    assert "## resnet18 arm A" in table and "## resnet18 arm B" in table
    assert "fewer than 4% of the train images" in table
    assert "bird, rodent, skunk" in table
    for header in ("accuracy, day", "macro-F1, night", "accuracy, rare", "macro-F1, common"):
        assert header in table
    with pytest.raises(ValueError):
        render_report({})
    with pytest.raises(ValueError):
        render_report({"arm A": []})


def test_split_predictions_refuse_inconsistent_input() -> None:
    with pytest.raises(ValueError, match="split must be"):
        _run("train", [OPOSSUM], [OPOSSUM], [True])
    with pytest.raises(ValueError, match="no images"):
        _run("val", [], [], [])
    with pytest.raises(ValueError, match="differ"):
        _run("val", [OPOSSUM, RABBIT], [OPOSSUM], [True, True])
    with pytest.raises(ValueError, match="features"):
        _run("val", [OPOSSUM, RABBIT], [OPOSSUM] * 2, [True] * 2, features=torch.zeros(3, 4))
    with pytest.raises(ValueError, match="unknown slice"):
        _run("val", [OPOSSUM], [OPOSSUM], [True]).in_slice("dusk")


# PCA


def test_pca_finds_known_components_with_a_fixed_sign() -> None:
    """Variance 9 along the first axis, 1 along the second, none along the third, shifted."""
    points = torch.tensor(
        [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    ) + torch.tensor([5.0, -2.0, 7.0])
    coords, explained = pca_2d(points)
    assert coords.dtype == torch.float64 and coords.shape == (4, 2)
    expected = torch.tensor([[3.0, 0.0], [-3.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
    torch.testing.assert_close(coords, expected, atol=1e-9, rtol=0)
    assert explained == pytest.approx((0.9, 0.1))
    again, _ = pca_2d(points.flip(0))
    torch.testing.assert_close(again, expected.flip(0), atol=1e-9, rtol=0)


def test_pca_refuses_too_few_images_features_or_variance() -> None:
    for shape in ((1, 5), (5, 1), (5,)):
        with pytest.raises(ValueError, match="at least 2"):
            pca_2d(torch.zeros(shape))
    with pytest.raises(ValueError, match="do not vary"):
        pca_2d(torch.ones(4, 3))


def test_project_pools_the_splits_in_order() -> None:
    cis = _run("val", [OPOSSUM, BIRD], [OPOSSUM] * 2, [True, False], [101, 102], torch.eye(2, 3))
    trans = _run("trans_val", [SKUNK], [OPOSSUM], [True], [201], torch.ones(1, 3))
    projection = project([cis, trans])
    assert projection.coords.shape == (3, 2)
    assert projection.labels == (OPOSSUM, BIRD, SKUNK)
    assert projection.locations == (101, 102, 201)
    assert projection.groups == ("cis", "cis", "trans")
    with pytest.raises(ValueError, match="no features kept for val"):
        project([dataclasses.replace(cis, features=None)])
    with pytest.raises(ValueError):
        project([])


# One pass over a split


def test_predict_split_returns_the_model_s_argmax_and_the_head_input(
    fixture: tuple[Split, Path],
) -> None:
    split, image_root = fixture
    records = split.records["trans_val"]
    torch.manual_seed(SEED)
    model = build_model("small_cnn", pretrained=False)
    assert isinstance(model, SmallCNN)
    run = predict_split(model, records, "trans_val", image_root, torch.device("cpu"), batch_size=8)
    assert run.split == "trans_val" and run.group == "trans"
    assert run.labels == tuple(r.label for r in records)
    assert run.night == tuple(r.night for r in records)
    assert run.locations == tuple(r.location for r in records)
    dataset = CCT20Dataset(records, image_root, EvalTransform())
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    with torch.no_grad():
        expected = model(images).argmax(dim=1).tolist()
        flat = model.conv_block_3(model.conv_block_2(model.conv_block_1(images))).flatten(1)
    assert list(run.predictions) == expected
    assert run.features is not None
    assert run.features.shape == (len(records), 32 * 28 * 28)
    torch.testing.assert_close(run.features, flat)
    assert not head_module(model)._forward_hooks

    lean = predict_split(
        model,
        records,
        "trans_val",
        image_root,
        torch.device("cpu"),
        batch_size=8,
        keep_features=False,
    )
    assert lean.features is None and lean.predictions == run.predictions


def test_predict_split_refuses_other_splits_and_unflagged_records(
    fixture: tuple[Split, Path],
) -> None:
    split, image_root = fixture
    model = build_model("small_cnn", pretrained=False)
    cpu = torch.device("cpu")
    with pytest.raises(ValueError, match="not from val"):
        predict_split(model, split.records["trans_val"], "val", image_root, cpu, batch_size=4)
    with pytest.raises(ValueError, match="split must be"):
        predict_split(model, split.records["train"], "train", image_root, cpu, batch_size=4)
    unflagged = [dataclasses.replace(r, night=None) for r in split.records["val"][:3]]
    with pytest.raises(ValueError, match="no day/night flag"):
        predict_split(model, unflagged, "val", image_root, cpu, batch_size=4)


# Figures


def test_write_feature_maps_writes_two_pngs(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(SEED)
    cis = _run(
        "val",
        [OPOSSUM, BIRD, SKUNK, RABBIT],
        [OPOSSUM] * 4,
        [True, False, True, False],
        [101, 102, 101, 102],
        torch.randn(4, 6, generator=generator),
    )
    trans = _run(
        "trans_val",
        [OPOSSUM, BIRD],
        [OPOSSUM] * 2,
        [True, False],
        [201, 201],
        torch.randn(2, 6, generator=generator),
    )
    projections = {"arm A": project([cis, trans]), "arm B": project([trans, cis])}
    paths = write_feature_maps(projections, tmp_path / "out")
    assert paths == (tmp_path / "out" / SPECIES_FIGURE, tmp_path / "out" / CAMERA_FIGURE)
    for path in paths:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > image.height  # the two arms sit side by side
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        (SPECIES_FIGURE, CAMERA_FIGURE)
    )
    with pytest.raises(ValueError):
        write_feature_maps({}, tmp_path)


# The command line


def _figures() -> list[Path]:
    return (
        sorted((ROOT / "docs" / "figures").rglob("*"))
        if (ROOT / "docs" / "figures").exists()
        else []
    )


def _table_rows(table: str) -> list[list[str]]:
    rows = []
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if line.startswith("| val |") or line.startswith("| trans_val |"):
            rows.append(cells)
    return rows


def test_the_dry_run_writes_the_table_and_two_pngs_under_out_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _figures()
    out = tmp_path / "out"
    assert slices.main(["--dry-run=True", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(
        (TABLE_NAME, SPECIES_FIGURE, CAMERA_FIGURE)
    )
    assert _figures() == before
    table = (out / TABLE_NAME).read_text(encoding="utf-8")
    assert table.count("## random weights") == 2
    rows = _table_rows(table)
    cut_rows = [r for r in rows if len(r) == 13]
    assert len(cut_rows) == 2 * 2 * 2  # two models, two cuts, val and trans_val
    for row in cut_rows:
        assert int(row[3]) + int(row[4]) == int(row[2])  # the slices add up to the split
    assert len([r for r in rows if len(r) == 9]) == 2  # one mix row per model

    printed = capsys.readouterr().out
    meta = load_metadata(*_fixture_paths(tmp_path / "ids"))
    names = {
        v for rs in meta.records.values() for r in rs for v in (r.image_id, r.file_name, r.seq_id)
    }
    assert names and not any(name in table or name in printed for name in names)


def _fixture_paths(directory: Path) -> tuple[Path, Path]:
    written = synthetic.write_synthetic_fixture(directory)
    return directory, written.paths["sidecar"]


@pytest.mark.parametrize("split", ["cis_test", "trans_test", "train"])
def test_the_command_line_refuses_every_split_but_val_and_trans_val(
    split: str, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as raised:
        slices.main(["--dry-run=True", "--split", split, "--out", str(tmp_path)])
    assert raised.value.code == 2
    assert not any(tmp_path.iterdir())


def test_the_command_line_reads_only_the_splits_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, set[str]]] = []
    real = slices.predict_split

    def spy(
        model: torch.nn.Module,
        records: tuple[Record, ...],
        split: str,
        *args: object,
        **kwargs: object,
    ) -> SplitPredictions:
        seen.append((split, {r.split for r in records}))
        return real(model, records, split, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(slices, "predict_split", spy)
    assert slices.main(["--dry-run=True", "--split", "trans_val", "--out", str(tmp_path)]) == 0
    assert seen == [("trans_val", {"trans_val"})] * 2


def _cache(tmp_path: Path) -> Path:
    """The fixture laid out as the 512 px cache: annotations/, images/ and the sidecar."""
    root = tmp_path / "cache"
    written = synthetic.write_synthetic_fixture(tmp_path / "fixture")
    annotations = root / CACHE_ANNOTATIONS
    annotations.mkdir(parents=True)
    for path in (tmp_path / "fixture").glob("*_annotations.json"):
        shutil.copy(path, annotations / path.name)
    shutil.copytree(written.image_root, root / CACHE_IMAGES)
    shutil.copy(written.paths["sidecar"], root / DAY_NIGHT_SIDECAR)
    return root


def _checkpoint(path: Path, name: str) -> Path:
    model = build_model(name, pretrained=False)
    torch.save({"model": name, "classes": list(CLASSES), "state_dict": model.state_dict()}, path)
    return path


def test_a_real_run_reads_the_cache_and_names_each_checkpoint(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    small = _checkpoint(tmp_path / "small.pt", "small_cnn")
    resnet = _checkpoint(tmp_path / "arm_a.pt", "resnet18")
    out = tmp_path / "out"
    # The real configs with no loader workers: spawning them dominates a run this small.
    config = tmp_path / "config.yaml"
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "  num_workers: 4 " in text and "  pin_memory: true\n" in text
    text = text.replace("  num_workers: 4 ", "  num_workers: 0 ")
    config.write_text(text.replace("  pin_memory: true\n", "  pin_memory: false\n"))
    argv = ["--checkpoint", "small_cnn", str(small), "--checkpoint", "resnet18_arm_a", str(resnet)]
    argv += ["--config", str(config), "--data-root", str(cache), "--out", str(out)]
    assert slices.main(argv) == 0
    table = (out / TABLE_NAME).read_text(encoding="utf-8")
    assert "## small_cnn" in table and "## resnet18 arm A" in table
    assert f"small_cnn: checkpoint sha256 {file_sha256(small)}" in table
    assert f"resnet18 arm A: checkpoint sha256 {file_sha256(resnet)}" in table
    assert "split hash" in table
    assert (out / SPECIES_FIGURE).exists() and (out / CAMERA_FIGURE).exists()


def test_the_command_line_refuses_incomplete_or_mixed_arguments(tmp_path: Path) -> None:
    out = str(tmp_path / "out")
    for argv in (
        ["--out", out],
        ["--checkpoint", "resnet18_arm_a", "a.pt", "--out", out],
        ["--data-root", ".", "--out", out],
        ["--dry-run=True", "--checkpoint", "resnet18_arm_a", "a.pt", "--out", out],
        ["--checkpoint", "dryrun", "a.pt", "--data-root", ".", "--out", out],
        [
            "--checkpoint",
            "resnet18_arm_a",
            "a.pt",
            "--checkpoint",
            "resnet18_arm_a",
            "b.pt",
            "--data-root",
            ".",
            "--out",
            out,
        ],
        ["--checkpoint", "no_such_config", "a.pt", "--data-root", ".", "--out", out],
    ):
        with pytest.raises(SystemExit) as raised:
            slices.main(argv)
        assert raised.value.code == 2
    assert not (tmp_path / "out").exists()
