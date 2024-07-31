import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageTk
import tkinter as tk

# Load the model and class labels
model = load_model('face_recognition_model.h5')
class_labels = np.load('class_labels.npy', allow_pickle=True).item()

# Define a threshold for prediction confidence
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for default camera

# Set up the Tkinter window
root = tk.Tk()
root.title("Face Recognition")
root.geometry("800x600")

# Create a label to display the video
label = tk.Label(root)
label.pack()

def update_frame():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Preprocess the image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize

    # Predict the class of the face
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    predicted_label = class_labels[predicted_class]

    # Determine if the confidence is below the threshold
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_label = "No face recognized"

    # Display the result on the frame
    frame = cv2.putText(frame, f"Recognized: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert frame to ImageTk
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=image)

    # Update the label with the new image
    label.config(image=photo)
    label.image = photo

    # Call update_frame again after 10 ms
    root.after(10, update_frame)

# Start the Tkinter event loop
update_frame()
root.mainloop()

# Release the camera
cap.release()
