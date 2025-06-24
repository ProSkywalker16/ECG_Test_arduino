ecg_model_weighted.tflite


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

# --- Load dataset ---
data = np.load('preprocessed_windows_balanced.npz')
X = data['X']
y = data['y']

# --- One-hot encode labels ---
num_classes = 6
y_cat = to_categorical(y, num_classes=num_classes)

# --- Compute class weights ---
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weights_dict = dict(enumerate(class_weights))
print("📊 Computed Class Weights:", class_weights_dict)

# --- Build the model ---
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(1800, 1)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),
    
    Conv1D(64, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Train the model with class weights ---
model.fit(X, y_cat,
          epochs=20,
          batch_size=32,
          validation_split=0.2,
          class_weight=class_weights_dict)

# --- Save the model ---
model.save("ecg_model_weighted.h5")

# --- Convert to TFLite (optional) ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("ecg_model_weighted.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Model saved and converted to TFLite.")
