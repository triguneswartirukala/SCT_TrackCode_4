import tensorflow as tf
import numpy as np
import cv2
import json
import sys

model = tf.keras.models.load_model('gesture_model.h5')

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Prediction: {class_names[class_index]} ({confidence * 100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
