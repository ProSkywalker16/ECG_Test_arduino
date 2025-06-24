import numpy as np
import pandas as pd
import tensorflow as tf

# === Class Mapping ===
CLASS_MAP = {
    0: "Normal Sinus Rhythm (NSR)",
    1: "Atrial Fibrillation (AFib)",
    2: "Ventricular Tachycardia (VT)",
    3: "Bradycardia",
    4: "ST Elevation",
    5: "Fusion of Ventricular and Normal Beat"
}

CLINICAL_GUIDANCE = {
    0: "Normal heart rhythm. No immediate action required.",
    1: "Irregular rhythm detected. May indicate AFib — clinical evaluation is recommended.",
    2: "Ventricular Tachycardia detected — potential emergency. Immediate cardiology review advised.",
    3: "Bradycardia detected. If symptomatic (e.g. dizziness, fatigue), seek evaluation.",
    4: "ST Elevation observed — possible myocardial infarction. Urgent action needed.",
    5: "Fusion beats detected — may be benign but monitor frequency and symptoms."
}

# === Load ECG signal from CSV ===
try:
    df = pd.read_csv("ecg_plot1.csv", header=None)
    signal = pd.to_numeric(df[0], errors='coerce').dropna().values.astype(np.float32)
    if signal.size == 0:
        raise ValueError("No valid numeric ECG data found in the file.")
except Exception as e:
    print(f"❌ Error reading ECG CSV: {e}")
    exit()

# === Preprocess signal ===
TARGET_LEN = 1800
if len(signal) > TARGET_LEN:
    start = (len(signal) - TARGET_LEN) // 2
    signal = signal[start:start + TARGET_LEN]
elif len(signal) < TARGET_LEN:
    signal = np.pad(signal, (0, TARGET_LEN - len(signal)), mode='constant')

signal = (signal - np.mean(signal)) / np.std(signal)
input_data = np.expand_dims(signal, axis=(0, -1)).astype(np.float32)  # shape: (1, 1800, 1)

# === Load TFLite Model ===
interpreter = tf.lite.Interpreter(model_path="ecg_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Perform inference ===
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

predicted_class = int(np.argmax(output))
confidence = float(np.max(output)) * 100

# === Output ===
print("\n📊 ECG Prediction on Arduino-Collected Signal")
if predicted_class not in CLASS_MAP:
    print(f"⚠️ Unknown predicted class: {predicted_class}")
else:
    print(f"🫀 Predicted Class: {CLASS_MAP[predicted_class]} ({predicted_class})")
    print(f"🔍 Confidence: {confidence:.2f}%")
    print("✅ Processed signal length:", len(signal))
    print("\n--- 🧠 Clinical Summary ---")
    print(CLINICAL_GUIDANCE[predicted_class])
