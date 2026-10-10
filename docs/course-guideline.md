# Course project guideline (as given by the class)

Pasted 2026-10-10 from the class handout. Wording kept as given; the lecture repo is
https://github.com/skillspire-github/pytorch-deep-learning (fork of mrdbourke's
"Learn PyTorch for Deep Learning: Zero to Mastery"; chapters 04 custom datasets, 06 transfer
learning and 09 deployment are the ones this project follows).

## Project Title: Image Classification Web Application

Objective: develop, train, and evaluate an end-to-end image classification model using PyTorch,
and deploy it as an interactive web application.

### 1. Task and dataset selection
- Project type: categorize the project explicitly as a Classifier (not a continuous
  regressor/predictor).
- Dataset: select an official dataset available directly through PyTorch's library
  (`torchvision.datasets`).
- Problem statement: clearly define the input data type, output classes, and the core challenge
  the model aims to solve.

### 2. Student-selected architecture and training
- Algorithm and architecture: choose an appropriate machine learning algorithm and model
  architecture for image classification. The only constraint is that the model must contain at
  least 2 layers.
- Loss function: select and implement a loss function that makes sense for your specific
  classification problem.
- Optimizer: choose an optimization algorithm (e.g., Adam, SGD) to train the network.
- Epochs and optimization: determine the optimal number of epochs to train the model,
  implementing strategies like early stopping or learning rate scheduling to prevent overfitting.
- Note: you must be prepared to justify your choices for the algorithm, architecture, loss
  function, and optimizer based on your chosen dataset.

### 3. Data pipeline
- Data loader: implement PyTorch's `DataLoader` to handle data batching, shuffling (for training),
  and preprocessing transformations.

### 4. Evaluation and inference tracking
- Performance evaluation: test the trained model on an unseen test set and report standard
  classification metrics.
- Inference timer: integrate a timing mechanism within your inference script to calculate and
  display the exact execution time (in seconds) for a single image prediction.

### 5. User interface and deployment
- Gradio web UI: build an intuitive web user interface using the Gradio library that allows users
  to upload an image, runs the inference function, and displays the predicted class label
  alongside the total inference time.
- Hugging Face Spaces deployment: deploy the complete application to Hugging Face Spaces,
  including the main application script (`app.py`), the saved model weights file, and a
  `requirements.txt` file.

## Note on the dataset rule

CCT20 (Caltech Camera Traps) is not in `torchvision.datasets`. The user decided to keep it: it is
loaded through a custom `torch.utils.data.Dataset` exactly as the course's chapter 04 teaches, and
everything downstream (DataLoader, training, evaluation, Gradio, Spaces) is identical. The TA's
reading of the rule has not been confirmed; mention the location-split motivation if asked.
