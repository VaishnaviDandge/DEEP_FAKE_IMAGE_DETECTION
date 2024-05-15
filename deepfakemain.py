import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def load_image(file_path):
    try:
        image = Image.open(file_path)
        image = image.resize((224, 224))  # Resize image to fit model input size
        image = np.array(image)           # Convert PIL image to numpy array
        image = preprocess_input(image)   # Preprocess image for the model
        return image
    except:
        return None

def predict_deepfake(file_path):
    image = load_image(file_path)
    if image is None:
        return None, "Error: Unable to process image"
    
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction, None

def classify_image(prediction):
    # The index of 'fake' class in ImageNet is 0, 'real' class is 283
    # You may need to adjust this based on the class indices in your dataset
    fake_probability = prediction[0, 0]
    real_probability = prediction[0, 283]

    if fake_probability > real_probability:
        return "Fake"
    else:
        return "Real"

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction, error = predict_deepfake(file_path)
        if error:
            label_result.config(text=error, fg="red")
            return
        
        # Classify the image as fake or real based on prediction
        result_text = classify_image(prediction)
        label_result.config(text=f"Prediction: {result_text}", fg="black")  
        
        # Display image
        image = Image.open(file_path)
        image = image.resize((300, 300))  # Resize image to display
        photo = ImageTk.PhotoImage(image)
        label_image.config(image=photo)
        label_image.image = photo

# GUI setup
root = tk.Tk()
root.title("Deepfake Detection using ResNet50")

# Styling
root.configure(bg="white")
root.geometry("500x500")

label_instruction = tk.Label(root, text="Please select an image to detect deepfake:", bg="white", font=("Arial", 12))
label_instruction.pack(pady=10)

button_browse = tk.Button(root, text="Browse", command=browse_image, bg="blue", fg="white", font=("Arial", 12))
button_browse.pack(pady=5)

label_image = tk.Label(root)
label_image.pack(pady=10)

label_result = tk.Label(root, text="", bg="white", font=("Arial", 12))
label_result.pack()

root.mainloop()
