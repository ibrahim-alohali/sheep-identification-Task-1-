# Sheep Image Classifier: Najdi, Harri, and Naeimi

A training project that classifies sheep images into three breeds found in Saudi Arabia: Najdi, Harri, and Naeimi. The original workflow used transfer learning in [Teachable Machine](https://teachablemachine.withgoogle.com/) and exported a Keras model for batch inference in Python.

## Repository contents

| File or folder | Purpose |
| --- | --- |
| [`keras_model.h5`](keras_model.h5) | Original exported model, with a 224 × 224 RGB input and three output scores. |
| [`labels.txt`](labels.txt) | Output-label order: `0 Najdi`, `1 Harri`, `2 Naeimi`. |
| [`Batch_test.py`](Batch_test.py) | Loads the model, evaluates `.jpg` images, and prints per-image and per-class results. |
| [`Najdi_test/`](Najdi_test/) | 11 included Najdi example images. |
| [`Harri_test/`](Harri_test/) | 10 included Harri example images. |
| [`Naeimi_test/`](Naeimi_test/) | 9 included Naeimi example images. |

Keep the model's output order aligned with `labels.txt`. Folder names must match a label followed by `_test`, including its spelling and case, because the script uses those names as dictionary keys.

## Historical environment and setup

The original documented environment was Python 3.9.x with TensorFlow 2.15.0, Pillow, and NumPy. The H5 file records Keras 2.4.0 export metadata. These are historical project details; compatibility with a newly installed environment has not been retested.

Create a virtual environment using a Python 3.9 interpreter:

```sh
python -m venv .venv
```

Activate it with the command for your shell:

```sh
# macOS or Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the historical dependency set:

```sh
python -m pip install tensorflow==2.15.0 pillow numpy
```

The script uses TensorFlow's bundled Keras API; it does not import the separate `keras` package. Availability of the historical TensorFlow build depends on the interpreter and platform.

## Batch evaluation

Run from the repository root so the relative model, label, and image paths resolve:

```sh
python Batch_test.py
```

For each directory ending in `_test`, the script processes filenames ending in `.jpg` without regard to extension case. PNG and `.jpeg` files are not included by the current filter.

Each image is converted to RGB, resized to 224 × 224 pixels, divided by 255 to produce values in `[0, 1]`, and passed to the model as a one-image batch. The highest output score determines the predicted label. The script prints each prediction and its score, then counts correct predictions against the folder label and prints a percentage per class. It does not write a report file.

## Limitations and review status

- The 30 included examples are a small, uneven set. The repository does not establish a held-out split or demonstrate that these images were excluded from training.
- The training dataset, class balance, and training history are not included. Generalization to new photos has not been established.
- The current script uses `[0, 1]` pixel scaling. Its match to the original Teachable Machine export preprocessing needs verification before using the printed results as performance evidence.
- A model score is not a calibrated guarantee. The evaluator counts only successfully processed images; decode or prediction errors are printed and excluded from the percentage denominator. A class with no processed images is displayed as 0%, which should not be interpreted as a measured accuracy.
- Filenames and model files are preserved from the original submission. IDE metadata was removed from version control, and Python caches and virtual environments are ignored.

This cleanup reviewed the source, model metadata, labels, and file counts. It did not load the model into TensorFlow, run inference, retrain it, or rerun the historical evaluation. No new accuracy claim is made.
