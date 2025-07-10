# Sheep Image Classifier (Najdi, Harri, Naeimi)

This repository contains a simple sheep breed image classification project. The model distinguishes between three types of sheep commonly found in Saudi Arabia: Najdi, Harri, and Naeimi. The workflow is based on transfer learning using [Teachable Machine](https://teachablemachine.withgoogle.com/) for training, with inference performed in Python using TensorFlow and Keras.

## Contents

- keras_model.h5: The exported trained model (Keras format, H5 file).
- labels.txt: Plaintext file listing class names in order.
- Batch_test.py: Batch script to evaluate images in test folders, reporting per-class results.
- Harri_test/, Naeimi_test/, Najdi_test/ : Each folder contains test images for that breed.
- .gitignore` , Excludes virtual environment and unnecessary files from version control.

## Requirements

- Python 3.9.x
- TensorFlow 2.15.0
- Pillow
- NumPy

Recommended to use a virtual environment.  
**Installation example:**
```sh
pip install tensorflow==2.15.0 pillow numpy
