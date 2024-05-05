# COVID-19 Classification Model

## Overview

This repository contains an AI model trained to classify chest X-ray images into three classes: normal, virus, and COVID-19. The model achieves an accuracy of 98% on the test dataset.

## Model Architecture

The model is built using transfer learning, specifically leveraging a pre-trained convolutional neural network (CNN) architecture. Transfer learning allows us to use the knowledge gained from training on a large dataset (typically ImageNet) and apply it to our specific classification task. We fine-tune the pre-trained model on our dataset containing chest X-ray images.

## Dataset

The dataset used for training and evaluation consists of chest X-ray images collected kaggle dataset. It contains images categorized into three classes: normal, virus, and COVID-19. The dataset is split into training, validation, and test sets to train and evaluate the model's performance.

## Training

During training, we use the transfer learning approach by initializing the model with pre-trained weights. We fine-tune the model on our dataset using techniques such as data augmentation and adjusting learning rates to improve generalization and achieve high accuracy.

## Evaluation

The model's performance is evaluated on a separate test dataset that was not seen during training or validation. We compute metrics such as accuracy, precision, recall, and F1-score to assess the model's ability to correctly classify chest X-ray images into normal, virus, or COVID-19 classes.

## Results

The model achieves an accuracy of 98% on the test dataset. Here are the detailed evaluation metrics:

Accuracy: 98%

# installation guide

1. create new conda environment

```bash
conda create -n covid python=3.12
```

2. activate the enviroment

```bash
conda activate covid
```

3. install the packages needed for the project

```bash
pip install -r requirements.txt
```
