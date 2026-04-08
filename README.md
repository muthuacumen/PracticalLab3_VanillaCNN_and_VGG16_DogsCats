# Vanilla CNN vs Fine-Tuned VGG16 -- Dogs vs Cats Classification

---

## Team Members

| Name | Student ID |
|---|---|
| Muthuraj Jayakumar | 9084570 |

---

## Overview

This lab demonstrates the core practice of Deep Learning Engineers: **don't train from scratch -- find an existing model that does something similar and fine-tune it for your specific task.**

We implement binary image classification (dogs vs cats) using two approaches and compare them head-to-head:

1. **Vanilla CNN** -- A 3-layer convolutional neural network trained from scratch
2. **Fine-Tuned VGG16** -- A pre-trained VGG16 model (ImageNet, 1.2M images) with the last 4 layers fine-tuned

The comparison covers accuracy, precision/recall/F1, PR curves, confusion matrices, and error analysis -- showing why transfer learning consistently outperforms training from scratch on small datasets.

**Course:** CSCN8010 -- Deep Learning

---

## Notebook

### `PracticalLab3_VanillaCNN_and_VGG16_DogsCats.ipynb`

| Section | Cells | What it covers |
|---|---|---|
| Introduction | 0 | Problem overview, two-model approach |
| Imports and Setup | 1--2 | Library imports, GPU/CPU detection |
| Data Loading | 3--5 | Load 5,000-image subset using `image_dataset_from_directory`, batch shape verification |
| EDA -- Sample Images | 6--7 | Image sanitization (corrupt/non-RGB removal), 4x4 grid of sample cats and dogs |
| EDA -- Class Distribution | 8--10 | Per-split class counts, bar chart confirming balanced dataset |
| EDA -- Image Size Analysis | 11--12 | Width/height distributions, aspect ratio scatter plot |
| EDA -- Pixel Intensity | 13--14 | Mean channel histograms, color channel distributions (RGB) |
| EDA -- Summary Statistics | 15--17 | Dataset summary table, EDA insights |
| Data Preprocessing | 17--18 | Augmentation pipeline: random flip, rotation (10%), zoom (20%) |
| Model 1: Vanilla CNN | 19--22 | 3-block CNN (32/64/128 filters), dropout 50%, 30 epochs, training curves |
| Model 2: Fine-Tuned VGG16 | 23--27 | VGG16 base (last 4 layers unfrozen), Dense(256), lr=1e-5, 30 epochs, training curves |
| Accuracy Comparison | 28--31 | Side-by-side bar chart, test accuracy for both models |
| Confusion Matrix | 32--33 | 2x2 heatmaps for both models |
| Precision, Recall, F1 | 34--36 | Classification reports, per-class metrics comparison table |
| PR Curve | 37--38 | Precision-Recall curves with AUC-PR for both models |
| Error Analysis | 39--41 | Misclassified image grids, per-class error breakdown |
| Conclusions | 42 | Key findings, practical takeaway, potential improvements |

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Asirra Dogs vs Cats (small subset) |
| **Source** | Kaggle / Microsoft Research |
| **Total images** | 5,000 |
| **Training** | 2,000 (1,000 cats + 1,000 dogs) |
| **Validation** | 1,000 (500 cats + 500 dogs) |
| **Test** | 2,000 (1,000 cats + 1,000 dogs) |
| **Input size** | Resized to 180x180 pixels |
| **Batch size** | 32 |

Download the dataset and place it in `data/kaggle_dogs_vs_cats_small/` with `train/`, `validation/`, and `test/` subdirectories, each containing `cat/` and `dog/` folders.

---

## Concepts Covered

### 1. Convolutional Neural Networks

| Component | What it does |
|---|---|
| **Conv2D** | Learns spatial filters (edges, textures, shapes) |
| **MaxPooling2D** | Downsamples feature maps, reduces computation |
| **Flatten + Dense** | Converts spatial features to classification logits |
| **Sigmoid output** | Produces probability for binary classification |

### 2. Regularization for Small Datasets

| Technique | What it does | Why it matters |
|---|---|---|
| **Data augmentation** | Random flip, rotation, zoom | Artificially increases training diversity |
| **Dropout (50%)** | Randomly zeros neurons during training | Prevents co-adaptation of features |
| **Early stopping** | Stops training when validation loss plateaus | Prevents overfitting to training data |

### 3. Transfer Learning and Fine-Tuning

| Strategy | Layers | Purpose |
|---|---|---|
| **Freeze early layers** | VGG16 layers 0--14 | Preserve universal features (edges, textures, colors) |
| **Unfreeze last 4 layers** | VGG16 layers 15--18 | Adapt task-specific features (cat ears, dog snouts) |
| **Low learning rate** | 1e-5 | Avoid destroying pre-learned representations |

> **Why VGG16?** ImageNet includes cats and dogs among its 1,000 categories. The model already knows what these animals look like -- we just need to sharpen its binary decision boundary.

### 4. Evaluation Metrics

| Metric | What it tells you |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | How much of what you predicted positive is actually positive |
| **Recall** | How much of the actual positives you captured |
| **F1-Score** | Harmonic mean of precision and recall |
| **PR Curve / AUC-PR** | Trade-off between precision and recall across thresholds |
| **Confusion Matrix** | Breakdown of TP, FP, FN, TN |

### 5. Connection to Modern AI

| This Lab | Modern Practice |
|---|---|
| Fine-tune VGG16 | Fine-tune ResNet, EfficientNet, ViT |
| ImageNet pre-training | Foundation models (CLIP, DINOv2) |
| Data augmentation | Advanced augmentation (CutMix, MixUp, RandAugment) |
| Binary classifier head | Multi-task / multi-label heads |

---

## How to Replicate

### Prerequisites

- Python 3.9 or higher
- Git

### Step 1 -- Clone the repository

```bash
git clone https://github.com/muthuacumen/PracticalLab3_VanillaCNN_and_VGG16_DogsCats.git
cd PracticalLab3_VanillaCNN_and_VGG16_DogsCats
```

### Step 2 -- Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 -- Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 -- Download the dataset

Download the Dogs vs Cats dataset and place it in `data/kaggle_dogs_vs_cats_small/` following the structure:

```
data/kaggle_dogs_vs_cats_small/
  train/
    cat/    (1,000 images)
    dog/    (1,000 images)
  validation/
    cat/    (500 images)
    dog/    (500 images)
  test/
    cat/    (1,000 images)
    dog/    (1,000 images)
```

### Step 5 -- Run the notebook

```bash
jupyter notebook PracticalLab3_VanillaCNN_and_VGG16_DogsCats.ipynb
```

Or open the folder in VS Code and use the built-in Jupyter extension.

> **Important**: Run cells **in order from top to bottom**. Later cells depend on variables defined in earlier cells.

---

## Dependencies

| Package | Version | Used for |
|---|---|---|
| `tensorflow` | 2.19.0 | Model building, training, data loading |
| `numpy` | 2.1.3 | Array operations |
| `pandas` | 2.3.1 | DataFrames for metric tables |
| `matplotlib` | 3.10.3 | All plots and visualizations |
| `seaborn` | latest | Heatmaps, styled charts |
| `scikit-learn` | latest | Classification reports, confusion matrix, PR curves |
| `Pillow` | latest | Image sanitization (corrupt file detection, RGB conversion) |
| `jinja2` | latest | Pandas styled DataFrames |
| `ipykernel` | 6.29.5 | Jupyter kernel |

---

## Key Findings

1. **Transfer learning significantly outperforms training from scratch.** Fine-tuned VGG16 achieves substantially higher test accuracy than the vanilla CNN, despite using the same 2,000 training images.

2. **Regularization is essential for small datasets.** Data augmentation and 50% dropout were critical -- without them, the vanilla CNN memorizes training data and fails to generalize.

3. **Fine-tuning strategy matters.** Freezing early layers (universal features) and unfreezing only the last 4 layers (task-specific features) with a low learning rate (1e-5) yields optimal performance.

4. **Error analysis reveals shared failure modes.** Both models struggle with cluttered backgrounds, unusual poses, partial views, and images containing multiple subjects.

5. **The vanilla CNN still learns meaningful patterns.** Despite its limitations, it significantly surpasses the 50% random baseline, proving that even simple architectures extract useful visual features.

---

## Potential Improvements

- Use the full 25,000-image dataset for better generalization
- Apply learning rate scheduling (ReduceLROnPlateau, cosine annealing)
- Try modern architectures: ResNet50, EfficientNetB0, or Vision Transformers
- Combine both models via ensemble methods
- Use test-time augmentation for more robust predictions

---

## Project Structure

```
PracticalLab3_CSCN8010/
  PracticalLab3_VanillaCNN_and_VGG16_DogsCats.ipynb  # Main notebook
  requirements.txt                                    # Python dependencies
  README.md                                           # This file
  .gitignore
  data/                                               # Dataset (not tracked in git)
    kaggle_dogs_vs_cats_small/
      train/
      validation/
      test/
  models/                                             # Saved checkpoints (not tracked)
    vanilla_cnn_best.keras
    vgg16_finetuned_best.keras
```
