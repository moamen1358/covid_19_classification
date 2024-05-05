from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import cv2
import numpy as np
import shutil
import os
import tensorflow as tf

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("covid_acc_98.keras")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Function to predict class label
def predict_class(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = ['Covid-19', 'Normal', 'Viral Pneumonia'][predicted_class_index]
    return predicted_class

# Function to save and process uploaded image
def save_and_process_image(file):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    processed_image_path = f"processed_{file.filename}"
    image = Image.open(file.filename)
    image.thumbnail((300, 300))
    image.save(processed_image_path)
    return processed_image_path

# Prediction endpoint
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        processed_image_path = save_and_process_image(image)
        class_label = predict_class(processed_image_path)
        os.remove(processed_image_path)
        return {"prediction": class_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
