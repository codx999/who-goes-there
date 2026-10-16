"""The Gradio demo (plan.md section 8; R04, R33, R35).

Upload a camera-trap photo and the app shows the top class with its confidence and the two
runners-up (`predict`), the forward pass's seconds, and, when the Grad-CAM box is ticked, the
top class's Grad-CAM overlay on the 224 x 224 frame the model saw. The overlay of an upload is
drawn in memory (`overlay_photo`) and never written to a file.
The app is built with Gradio's analytics off: no version check and no telemetry, so a local
build or launch never calls `api.gradio.app`. `--share` does, by design: Gradio asks it once for a
tunnel server (and may download its tunnel client), and writes `.gradio/certificate.pem` into the
working directory, which `.gitignore` covers.

Examples come from the repo's `assets/` folder only (`example_photos`), the trans-test photos the
user picks (item 8.05); each goes through `assets_image`, so a link out of the folder is refused.
The folder may be empty or missing: the app is then built without examples. Example photos never
come from `data/`.

    python -m trailcam.demo.app --checkpoint <run>.pt --config-name resnet18_arm_b
    python -m trailcam.demo.app --checkpoint <run>.pt --share
    python -m trailcam.demo.app --check

`--check` builds the interface and exits without launching; without `--checkpoint` it builds the
config's model with random weights, so it needs no trained run. `--share` is the demo-day backup
if the Space is down: Gradio's `share=True`, a temporary public link served from this machine,
which lasts only while it runs. The default is local only; `--check` accepts `--share` and opens
no tunnel.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import gradio as gr
from PIL import Image
from torch import nn

from trailcam.demo.predict import TOP_K, load_model, predict
from trailcam.evaluate.gradcam import ASSETS_DIR, ZeroMapError, assets_image, overlay_photo
from trailcam.model.factory import build_model
from trailcam.train.config import CONFIG_PATH, load_config

DEFAULT_CONFIG: Final = "resnet18_arm_b"  # both arms are a ResNet-18; open question 7 picks one
PHOTO_SUFFIXES: Final = (".jpg", ".jpeg", ".png")
CITATION: Final = (
    "Photos: Caltech Camera Traps (CCT20). Sara Beery, Grant Van Horn, Pietro Perona, "
    '"Recognition in Terra Incognita", ECCV 2018. Data licence CDLA-Permissive 1.0.'
)


def example_photos(assets_root: Path = ASSETS_DIR) -> list[Path]:
    """The photos directly under `assets_root`, sorted; none if the folder is missing. Each is
    checked by `assets_image`, which refuses a link that leaves the folder."""
    root = Path(assets_root)
    if not root.is_dir():
        return []
    photos = [p for p in root.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES]
    return sorted(assets_image(photo, root) for photo in photos)


def classify(
    model: nn.Module, photo: Image.Image | None, show_gradcam: bool
) -> tuple[dict[str, float], str, Image.Image | None]:
    """The app's handler: the top-3 scores for `gr.Label`, the seconds as text, and the Grad-CAM
    overlay or None. Runs outside `torch.inference_mode()`, which Grad-CAM's autograd needs."""
    if photo is None:
        raise gr.Error("Upload a camera-trap photo first.")
    result = predict(model, photo)
    scores = dict(result.top)
    seconds = f"{result.seconds:.3f} s (forward pass only)"
    if not show_gradcam:
        return scores, seconds, None
    try:
        return scores, seconds, overlay_photo(model, photo).image
    except ZeroMapError:
        gr.Warning("The Grad-CAM map of this photo is zero everywhere: no overlay to show.")
        return scores, seconds, None


def build_app(model: nn.Module, examples: Sequence[Path] = ()) -> gr.Blocks:
    """The interface around `model`; `examples` are `assets/` photos, and may be empty."""
    # No telemetry: with analytics off, Gradio skips its version check and analytics calls too.
    app = gr.Blocks(title="Who Goes There", analytics_enabled=False)
    with app:
        gr.Markdown(
            "# Who Goes There\nA camera-trap species classifier: 11 animals and `car`, trained "
            "on Caltech Camera Traps."
        )
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", label="Camera-trap photo")
                show_gradcam = gr.Checkbox(label="Show Grad-CAM of the top class", value=False)
                run = gr.Button("Classify", variant="primary")
            with gr.Column():
                scores = gr.Label(
                    num_top_classes=TOP_K, label="Top class and two runners-up (model confidence)"
                )
                seconds = gr.Textbox(label="Inference time", interactive=False)
                heatmap = gr.Image(type="pil", label="Grad-CAM (224 x 224 frame)")
        if examples:
            gr.Examples(examples=[[str(path)] for path in examples], inputs=[photo])
        gr.Markdown(CITATION)

        def handle(
            image: Image.Image | None, gradcam: bool
        ) -> tuple[dict[str, float], str, Image.Image | None]:
            return classify(model, image, gradcam)

        run.click(handle, inputs=[photo, show_gradcam], outputs=[scores, seconds, heatmap])
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trailcam.demo.app", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="a train.py checkpoint")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG, help="the checkpoint's config")
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--assets", type=Path, default=ASSETS_DIR, help="the examples' folder")
    parser.add_argument("--check", action="store_true", help="build the interface, then exit")
    parser.add_argument("--share", action="store_true", help="a temporary public Gradio link")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        if not args.check:
            parser.error("--checkpoint is required unless --check is given")
        model = build_model(load_config(args.config_name, args.config_path).model, pretrained=False)
        model.eval()
    else:
        model = load_model(args.checkpoint, args.config_name, args.config_path)
    examples = example_photos(args.assets)
    app = build_app(model, examples)
    if args.check:
        print(f"interface built, not launched: {len(examples)} examples")
        return 0
    app.launch(share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
