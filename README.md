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

4. Start FastAPI Application

```bash
uvicorn main:app --reload
```

5. Access API Endpoints in Postman:
   Open Postman.
   Create a new request.
   Set the request method ( POST) and enter the URL of your FastAPI application along with the specific endpoint you want to access.

## Usage

### Using Postman to Upload an Image

1. Open Postman.
2. Create a new request.
3. Set the request method to POST.
4. Enter the URL of your FastAPI application along with the specific endpoint that expects the image.
   - Example URL: `http://localhost:8000/predict`
5. Click on the "Body" tab.
6. Choose "form-data" as the body type.
7. Add a key-value pair where the key corresponds to the name of the parameter expected by your API for the image, and the value is the image file you want to upload.
   - Key: `image`
   - Value: [Choose Files] button to select the image file from your local system.
8. Click the "Send" button to send the request to your FastAPI application.
