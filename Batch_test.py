import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)  # Teachable Machine default

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Only test folders ending with _test
test_root = "."

results = []

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Build per-class stats
summary = {label: {"total": 0, "correct": 0, "details": []} for label in labels}

for class_folder in os.listdir(test_root):
    if not class_folder.endswith("_test"):
        continue
    folder_path = os.path.join(test_root, class_folder)
    if not os.path.isdir(folder_path):
        continue
    true_class = class_folder.replace("_test", "")
    print(f"\nTesting folder: {class_folder}")

    num_correct = 0
    num_total = 0

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg")):
            continue
        img_path = os.path.join(folder_path, fname)
        try:
            img_array = preprocess_image(img_path)
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            pred_label = labels[pred_idx]
            correct = (pred_label.lower() == true_class.lower())
            summary[true_class]["details"].append((fname, pred_label, float(preds[0][pred_idx])))
            if correct:
                num_correct += 1
            num_total += 1
            print(f"{fname}: Predicted {pred_label} ({preds[0][pred_idx]:.2f}) - {'CORRECT' if correct else 'WRONG'}")
        except Exception as e:
            print(f"{fname}: ERROR - {e}")

    summary[true_class]["total"] = num_total
    summary[true_class]["correct"] = num_correct

# Print summary
print("\n=== Batch Test Results ===")
for label in labels:
    total = summary[label]["total"]
    correct = summary[label]["correct"]
    acc = correct / total * 100 if total > 0 else 0
    print(f"{label}: {correct}/{total} correct ({acc:.1f}%)")
