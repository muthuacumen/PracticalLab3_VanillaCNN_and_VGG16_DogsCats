# Practical Lab 3 — Vanilla CNN and Fine-Tune VGG16 for Dogs vs Cats Classification

**Course:** CSCN8010 — Deep Learning  
**Objective:** Demonstrate the practice of Deep Learning Engineers — take an existing model that does something similar, and fine-tune it for the specific task at hand.

## Problem Statement

Binary image classification: distinguish between photos of dogs and cats using two approaches:

1. **Vanilla CNN** — A convolutional neural network trained from scratch
2. **Fine-Tuned VGG16** — A pre-trained VGG16 model (ImageNet weights) fine-tuned for this task

## Dataset

The **Asirra Dogs vs Cats** dataset (5,000-image subset):

| Split | Images | Per Class |
|-------|--------|-----------|
| Training | 2,000 | 1,000 cats + 1,000 dogs |
| Validation | 1,000 | 500 cats + 500 dogs |
| Test | 2,000 | 1,000 cats + 1,000 dogs |

## Project Structure

```
PracticalLab3_CSCN8010/
├── PracticalLab3_VanillaCNN_and_VGG16_DogsCats.ipynb  # Main notebook
├── TeachMuthu.md                                       # Personal study guide
├── requirements.txt                                    # Python dependencies
├── README.md                                           # This file
├── .gitignore
├── data/                                               # Dataset (not tracked in git)
│   ├── kaggle_dogs_vs_cats_small/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── kagglecatsanddogs_5340/
└── models/                                             # Saved model checkpoints (not tracked)
    ├── vanilla_cnn_best.keras
    └── vgg16_finetuned_best.keras
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/muthuacumen/PracticalLab3_VanillaCNN_and_VGG16_DogsCats.git
   cd PracticalLab3_VanillaCNN_and_VGG16_DogsCats
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the Dogs vs Cats dataset and place it in the `data/` directory following the structure above.

5. Open and run the notebook:
   ```bash
   jupyter notebook PracticalLab3_VanillaCNN_and_VGG16_DogsCats.ipynb
   ```

## Notebook Outline

1. **Introduction** — Problem overview and approach
2. **Data Loading** — Load the 5,000-image subset
3. **EDA** — Exploratory data analysis with visualizations
4. **Data Preprocessing** — Augmentation pipeline
5. **Model 1: Vanilla CNN** — Train from scratch with augmentation and dropout
6. **Model 2: Fine-Tuned VGG16** — Transfer learning with fine-tuning
7. **Model Comparison** — Accuracy, confusion matrix, precision/recall/F1, PR curves, error analysis
8. **Conclusions** — Findings and takeaways

## Key Findings

- Transfer learning with VGG16 significantly outperforms training from scratch
- Fine-tuning the last few layers of a pre-trained model is the standard practice for Deep Learning Engineers
- Data augmentation and dropout are essential regularization techniques for small datasets
