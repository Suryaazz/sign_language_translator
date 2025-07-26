import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Path to dataset
dataset_path = 'dataset/train/'

# Image settings
IMG_SIZE = 64

# Prepare data lists
images = []
labels = []

# Load and preprocess images (limit to 300 per class)
for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)
    if not os.path.isdir(folder_path):
        continue

    count = 0
    for img_file in os.listdir(folder_path):
        if count >= 300:  # ✅ Limit images per class to avoid memory issues
            break
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)
        count += 1

print(f"✅ Loaded {len(images)} images.")

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Encode labels (A → 0, B → 1, ..., Z → 25)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training and validation sets (80% / 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Validation samples: {len(X_val)}")
print(f"✅ Classes: {le.classes_}")






from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, num_classes=29)
y_val_cat = to_categorical(y_val, num_classes=29)

# Build CNN model
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Helps prevent overfitting
model.add(Dense(29, activation='softmax'))  # 29 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("🔁 Training model...")

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# Save the model
model.save("model/asl_cnn_model.h5")
print("✅ Model trained and saved as 'model/asl_cnn_model.h5'")
