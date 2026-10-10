# Who Goes There

A PyTorch species classifier for camera-trap photos, tested on camera locations it never saw
during training. Bootcamp capstone; presentation 2026-10-26.

Given one camera-trap image, the model names the animal in it. The headline result compares
accuracy on **cis** test images (cameras also seen in training) with **trans** test images
(cameras never seen), which shows how much a model leans on a camera's fixed background instead
of the animal.

## Status

Repo bootstrapped (tooling, data guard, CI). Design document and training code follow; see
`docs/` for the plan, the data inspection and the progress log.

## Dataset

[Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps), CCT20 benchmark
subset, from LILA BC. About 58k images from 20 camera locations in the American Southwest, with
the official location-based train / cis test / trans test split.

The images are **not** in this repo and never will be: `scripts/guard_data.py`, git hooks and CI
refuse them. Training downloads them from LILA inside a Kaggle notebook.

License: [CDLA-Permissive 1.0](https://cdla.dev/permissive-1-0/). Please cite:

> Sara Beery, Grant Van Horn, Pietro Perona. "Recognition in Terra Incognita."
> Proceedings of the European Conference on Computer Vision (ECCV), 2018.
> https://arxiv.org/abs/1807.04975

## Development

Python 3.12 managed by [uv](https://docs.astral.sh/uv/). One script runs everything CI runs:

```bash
bash scripts/check.sh
```

Enable the data-guard hooks once per clone:

```bash
git config core.hooksPath .githooks
```

## License

Code is MIT licensed (see `LICENSE`). The dataset has its own license, above.
