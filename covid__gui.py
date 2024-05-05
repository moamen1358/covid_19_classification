import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

# Global variables
selected_image = None
file_path = None
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

# Function to load and display the image
def load_image():
    global selected_image, file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        selected_image = Image.open(file_path)
        selected_image.thumbnail((300, 300))  # Resize image to fit in GUI
        photo = ImageTk.PhotoImage(selected_image)
        label.config(image=photo)
        label.image = photo  # Keep a reference to the image to prevent garbage collection
        predict_button.config(state=tk.NORMAL)  # Enable predict button
        
        # Save the image in the same directory as the code
        code_directory = os.path.dirname(os.path.abspath(__file__))
        image_filename = os.path.basename(file_path)
        save_path = os.path.join(code_directory, image_filename)
        selected_image.save(save_path)
        print(f"Image saved at: {save_path}")

# Function to predict and display the result
def predict_image():
    global file_path
    if selected_image and file_path:
        class_label = predict_class(file_path)
        prediction_label.config(text=f"Prediction: {class_label}", font=("Helvetica", 14, "bold"), fg="#009933")

# Create tkinter window
root = tk.Tk()
root.title("Medical Image Classifier")
root.configure(bg="#e6f5ff")  # Light blue background

# Set width and height of the window
window_width = 450
window_height = 500
root.geometry(f"{window_width}x{window_height}")

# Create a label to display the image
label = tk.Label(root, text="Upload an X-ray Image", font=("Helvetica", 16, "bold"), bg="#156511")  # Light blue background
label.pack(expand=True, fill='both', padx=10, pady=10)

# Create a button to select an image file
select_button = tk.Button(root, text="Select Image", font=("Helvetica", 12), command=load_image, bg="#66ccff", fg="white")
select_button.pack(pady=10)

# Create a button to predict
predict_button = tk.Button(root, text="Predict", font=("Helvetica", 12), command=predict_image, state=tk.DISABLED, bg="#009933", fg="white")
predict_button.pack(pady=10)

# Create a label to display prediction result
prediction_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#e6f5ff")  # Light blue background
prediction_label.pack(pady=10)

# Run the tkinter event loop
root.mainloop()
