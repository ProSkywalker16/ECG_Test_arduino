import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the processed data
data = np.load("preprocessed_windows_balanced.npz")
X = data["X"]
y = data["y"]

# Normalize input data to range [0, 1]
X = X / 1023.0
X = X[..., np.newaxis]  # Add channel dimension

# One-hot encode the labels
num_classes = 6;
y_cat = to_categorical(y, num_classes=num_classes)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

# Define the model
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32)

# Save model
model.save("ecg_model.h5")
print("✅ Saved ecg_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("ecg_model.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Converted and saved ecg_model.tflite")
