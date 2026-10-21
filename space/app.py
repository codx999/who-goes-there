"""The Hugging Face Space (plan.md section 8; item 11.01; R33 to R36).

The Space runs this file as `app.py` at its root, beside the `trailcam` package, `config.yaml`, the
`assets/` examples, the weights (`model.pt`) and the name of their config (`config_name.txt`, item
13.11; since item 13.15 phase 3's finalist, `p3_convnext_cp`), all uploaded by
`scripts/deploy_space.py`. It holds no prediction code of its own: the interface is `build_app`
of `trailcam/demo/app.py`, and the button runs `classify`, which calls `predict` of
`trailcam/demo/predict.py` (eval transform,
`timed_forward`, softmax top 3) and, when the box is ticked, Grad-CAM. Grad-CAM is in the Space and
off by default (open question 5, closed 2026-10-16). Analytics are off, as in the local app.

ZeroGPU-ready: the same file runs on CPU Basic now and on ZeroGPU later, with only the Space's
hardware setting changed (and the requirements' torch, plan.md section 8).

- `spaces` is imported first, before torch, if it is there (on Hugging Face), else the handler
  is left as it is (locally). Off ZeroGPU, `spaces.GPU` has no effect.
- The model goes to `cuda` here, at module level, when CUDA is available, else it stays on the
  CPU. On ZeroGPU, CUDA is emulated outside `spaces.GPU` calls, and Hugging Face asks for models to
  be placed on `cuda` at startup, not inside the call.
- The one GPU-dependent function, the button's handler (prediction and Grad-CAM, which needs
  autograd), is wrapped in `spaces.GPU`. It takes the photo and the switch only and reads the
  model from this module, so a ZeroGPU call never has to pass the model.

Run locally with the chosen run's checkpoint copied to `space/model.pt` (ignored by git):

    uv run python space/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# Before torch, as Hugging Face asks: on ZeroGPU, `spaces` sets up torch's CUDA emulation.
try:
    import spaces  # type: ignore[import-not-found]  # on Hugging Face only, not in the lock
except ImportError:
    spaces = None

import torch
from PIL import Image

from trailcam.demo.app import Handler, Result, build_app, classify, example_photos
from trailcam.demo.predict import SPACE_CONFIG_FILE, load_model, read_config_name
from trailcam.evaluate.gradcam import ASSETS_DIR
from trailcam.train.config import CONFIG_PATH

# Beside this file: the Space's root, or `space/` in a checkout. CONFIG_PATH and ASSETS_DIR are
# found from the package, so they are the Space's root files there and the repo's root ones here.
WEIGHTS: Final = Path(__file__).resolve().parent / "model.pt"
CONFIG_NAME: Final = read_config_name(Path(__file__).resolve().parent / SPACE_CONFIG_FILE)
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu(handler: Handler) -> Handler:
    """`spaces.GPU(handler)` where `spaces` is installed (a GPU for each call on ZeroGPU, no effect
    on CPU Basic), else `handler` unchanged."""
    if spaces is None:
        return handler
    wrapped: Handler = spaces.GPU(handler)
    return wrapped


MODEL: Final = load_model(WEIGHTS, CONFIG_NAME, CONFIG_PATH).to(DEVICE)


@gpu
def classify_on_device(photo: Image.Image | None, show_gradcam: bool) -> Result:
    """The button's handler: `classify` on the Space's model."""
    return classify(MODEL, photo, show_gradcam)


demo: Final = build_app(MODEL, example_photos(ASSETS_DIR), classify_on_device)

if __name__ == "__main__":
    demo.launch()
