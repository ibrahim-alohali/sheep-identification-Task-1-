# Sheep Breed Classifier: Najdi, Harri and Naeimi

An image-classification exercise from my AI and robotics training at Smart Methods (summer 2025). I trained a model in [Teachable Machine](https://teachablemachine.withgoogle.com/) (transfer learning) to tell apart three sheep breeds common in Saudi Arabia, exported it as a Keras model and wrote a Python script that tests it on folders of images.

## What's in the repository

| File or folder | Contents |
| --- | --- |
| [`keras_model.h5`](keras_model.h5) | The exported model: 224 × 224 RGB input, three softmax outputs. |
| [`labels.txt`](labels.txt) | Class order: `0 Najdi`, `1 Harri`, `2 Naeimi`. |
| [`Batch_test.py`](Batch_test.py) | Runs the model on every test folder and prints per-image and per-class results. |
| [`Najdi_test/`](Najdi_test/), [`Harri_test/`](Harri_test/), [`Naeimi_test/`](Naeimi_test/) | Test images: 11 Najdi, 10 Harri and 9 Naeimi. |

## Setup

The project used Python 3.9 with TensorFlow 2.15.0, Pillow and NumPy. The model file itself was saved by Keras 2.4.0.

```sh
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install tensorflow==2.15.0 pillow numpy
```

The script uses the Keras API bundled with TensorFlow, so the separate `keras` package isn't needed.

## Run the evaluation

From the repository root:

```sh
python Batch_test.py
```

The script goes through every folder whose name ends in `_test`, takes the part before `_test` as the true breed, and classifies each `.jpg` image in it. Each image is converted to RGB, resized to 224 × 224, scaled to [0, 1] and passed to the model; the class with the highest score is the prediction. It prints every prediction with its score, then the number and percentage of correct predictions per breed.

Folder names must match the labels in `labels.txt` exactly, including capital letters. Only `.jpg` files are read.

## Known issues

- Teachable Machine's own export code centre-crops each image and scales the pixels to [−1, 1]. `Batch_test.py` stretches the image and scales to [0, 1] instead, so the accuracy it prints is not a reliable measure of the model.
- The test set is small (30 images), and nothing here shows whether these images were kept out of training. The training images and training history aren't included.
- An image that fails to load is reported and left out of the percentage. A breed with no processed images shows 0%.
