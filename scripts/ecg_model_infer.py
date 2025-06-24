import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="ecg_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (based on your 5-class setup)
CLASS_LABELS = {
    0: "Normal Sinus Rhythm (NSR)",
    1: "Atrial Fibrillation (AFib)",
    2: "Ventricular Tachycardia (VT)",
    3: "Bradycardia",
    4: "ST Elevation"
}

# Load preprocessed ECG samples
with np.load("processed_windows.npz") as data:
    X = data["X"]
    y = data["y"]

# Choose a sample (can iterate over more)
sample_index = 0
sample_input = X[sample_index].reshape(1, -1, 1).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], sample_input)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

# Decode prediction
predicted_class = int(np.argmax(prediction))
confidence = float(np.max(prediction))

print(f"🫀 Predicted Class: {CLASS_LABELS[predicted_class]} ({predicted_class})")
print(f"🔍 Confidence: {confidence * 100:.2f}%")
print(f"✅ Actual Label: {CLASS_LABELS[y[sample_index]]} ({y[sample_index]})")
