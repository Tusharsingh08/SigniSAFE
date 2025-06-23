import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# Preprocess Signature Image
def preprocess_signature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Resize to 128x128
    img = np.expand_dims(img, axis=-1)  # Add channel dimension -> (128, 128, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension -> (1, 128, 128, 1)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Build CNN Model
def build_signature_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),  # Explicitly define input shape
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and Preprocess Training Data
def load_data(data_dir):
    images = []
    labels = []
    
    for category in ['genuine', 'forged']:
        category_path = os.path.join(data_dir, category)
        
        # Debugging: Check if directory exists
        if not os.path.exists(category_path):
            print(f"[ERROR] Directory not found: {category_path}")
            continue
        
        label = 1 if category == 'genuine' else 0
        
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            
            # Debugging: Print image paths
            print(f"Loading image: {img_path}")
            
            # Read and preprocess image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Could not read image: {img_path}")
                continue
            
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=-1)  # Add channel
            img = np.expand_dims(img, axis=0)   # Add batch dimension
            img = img / 255.0  # Normalize
            
            images.append(img)
            labels.append(label)
    
    # Handle empty data
    if len(images) == 0:
        raise ValueError("[ERROR] No images were loaded. Check dataset path.")
    
    images = np.vstack(images)  # Stack to (batch_size, 128, 128, 1)
    labels = np.array(labels)
    
    print(f"Total images loaded: {len(images)}")
    return images, labels
    # Convert lists to numpy arrays
    images = np.vstack(images)  # Stack to (batch_size, 128, 128, 1)
    labels = np.array(labels)
    return images, labels


# Train the Model
data_dir = 'dataset/signatures'
train_images, train_labels = load_data(data_dir)
model = build_signature_model()
model.fit(train_images, train_labels, epochs=10)

# Save the Model
model.save('models/signature_model.h5')
