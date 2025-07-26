import os
import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model
model = load_model("model/asl_cnn_model.h5")

# Labels used in training
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# ✅ This must be BEFORE any usage of cap
cap = cv2.VideoCapture(0)

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Image size for model
IMG_SIZE = 64

# Variables for word-building logic
output_text = ""
previous_label = ""
same_count = 0
threshold = 15
max_text_len = 20

print("✅ ASL Translator Running")
print("🟢 Show gestures in the box.")
print("🔊 Press 's' to speak | 🧹 Press 'c' to clear | ❌ Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Region of Interest (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocessing
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_array = img_to_array(roi_rgb) / 255.0
    roi_array = np.expand_dims(roi_array, axis=0)

    # Prediction
    prediction = model.predict(roi_array, verbose=0)
    pred_index = np.argmax(prediction)
    pred_label = labels[pred_index]
    confidence = prediction[0][pred_index]

    # Word building logic
    if confidence > 0.9 and pred_label != 'nothing':
        if pred_label == previous_label:
            same_count += 1
        else:
            same_count = 0
        previous_label = pred_label

        if same_count > threshold:
            if pred_label == 'space':
                output_text += " "
            elif pred_label == 'del':
                output_text = output_text[:-1]
            else:
                if len(output_text) < max_text_len:
                    output_text += pred_label
            same_count = 0
    else:
        same_count = 0

    # Draw results
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, f"{pred_label} ({confidence:.2f})",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(frame, f"Text: {output_text}",
                (10, 400), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2)

    cv2.imshow("ASL Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        output_text = ""
    elif key == ord('s') and output_text.strip():
        engine.say(output_text)
        engine.runAndWait()

cap.release()
cv2.destroyAllWindows()

