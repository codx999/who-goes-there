# Course project requirements, and where this project meets them

The capstone for *AI and Machine Learning: Code, Train, & Deploy*. The course follows the public
lecture repo <https://github.com/skillspire-github/pytorch-deep-learning>, a fork of Daniel
Bourke's *Learn PyTorch for Deep Learning: Zero to Mastery*; this project builds on its chapters
04 (custom datasets), 06 (transfer learning) and 09 (deployment). Below, the requirements in my
own words, each with where the project meets it.

| Requirement, in short | Where it is met |
| --- | --- |
| A classifier, with a clear input, output classes and core challenge | One camera-trap photo in, one of 12 classes out; the challenge is cameras never seen in training (README) |
| A dataset from `torchvision.datasets` | CCT20 is not in it; loaded through a custom `torch.utils.data.Dataset`, as chapter 04 teaches (see the note below) |
| A model of at least 2 layers, with a justified architecture | A small CNN, ResNet-18 and the final ConvNeXt-Tiny, all ImageNet-pretrained except the small CNN (`trailcam/model/`) |
| A loss function and an optimizer that fit the problem | Cross-entropy; AdamW with a slower rate for the pretrained body (`config.yaml`) |
| A chosen number of epochs, with early stopping or scheduling | At most 10 epochs, cosine schedule with warm-up, early stopping on validation macro-F1 |
| A `DataLoader` with batching, shuffling and transforms | `trailcam/data/loaders.py` |
| Evaluation on an unseen test set with standard metrics | Accuracy and macro-F1 on familiar and on new cameras (README, results) |
| An inference timer for a single image, in seconds | `trailcam/demo/predict.py`, shown in the app |
| A Gradio UI showing the class and the inference time | `trailcam/demo/app.py` |
| Deployment to Hugging Face Spaces with `app.py`, weights and `requirements.txt` | `space/`, live at <https://huggingface.co/spaces/Codx999/who-goes-there> |

## Note on the dataset rule

CCT20 (Caltech Camera Traps) is not in `torchvision.datasets`. It was kept because it is the
dataset with a published split by camera, which is what makes "cameras never seen" measurable.
It is loaded through a custom `Dataset`, and everything downstream (DataLoader, training,
evaluation, Gradio, Spaces) follows the course.
