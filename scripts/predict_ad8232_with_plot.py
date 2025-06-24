import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# 🩺 Class decoding for our 6-class model
CLASS_MAP = {
    0: "Normal Sinus Rhythm (NSR)",
    1: "Atrial Fibrillation (AFib)",
    2: "Ventricular Tachycardia (VT)",
    3: "Bradycardia",
    4: "ST Elevation",
    5: "Fusion of Ventricular and Normal Beat"
}

# === Load ECG signal from CSV ===
try:
    df = pd.read_csv("ecg_arrhythmia_sim.csv", header=None)

    # Detect if multiple columns are present (e.g., time + voltage)
    if df.shape[1] > 1:
        print("📌 Detected multiple columns — using second column as ECG data.")
        signal = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    else:
        signal = pd.to_numeric(df.iloc[:, 0], errors='coerce')

    signal = signal.dropna().values.astype(np.float32)

    if signal.size == 0:
        raise ValueError("No valid numeric ECG data found in the file.")
    
    print(f"✅ Loaded {len(signal)} ECG samples.")

except Exception as e:
    print(f"❌ Error reading ECG CSV: {e}")
    exit()

# === Preprocess signal ===
TARGET_LEN = 1800  # must match model input
if len(signal) > TARGET_LEN:
    start = (len(signal) - TARGET_LEN) // 2
    signal = signal[start:start + TARGET_LEN]
elif len(signal) < TARGET_LEN:
    signal = np.pad(signal, (0, TARGET_LEN - len(signal)), mode='constant')

# Keep a copy before normalization for plotting
plot_signal = signal.copy()

# Normalize signal (z-score)
signal = (signal - np.mean(signal)) / np.std(signal)
input_data = np.expand_dims(signal, axis=(0, -1)).astype(np.float32)  # (1, 1800, 1)

# === Load and run model ===
interpreter = tf.lite.Interpreter(model_path="ecg_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# === Decode result ===
predicted_class = int(np.argmax(output))
confidence = float(np.max(output)) * 100
class_name = CLASS_MAP.get(predicted_class, "Unknown")

# === Show prediction ===
print("\n📊 ECG Prediction on Arduino-Collected Signal")
print(f"🫀 Predicted Class: {class_name} ({predicted_class})")
print(f"🔍 Confidence: {confidence:.2f}%")
print(f"✅ Processed signal length: {len(signal)}")

# === 🧠 Clinical Summary ===
print("\n--- 🧠 Clinical Summary ---")
if predicted_class == 0:
    print("The ECG appears normal with regular sinus rhythm.")
elif predicted_class == 1:
    print("The ECG indicates signs of Atrial Fibrillation (AFib), an arrhythmia originating in the atria. Clinical evaluation is recommended.")
elif predicted_class == 2:
    print("The ECG suggests Ventricular Tachycardia (VT), a potentially life-threatening condition. Immediate medical attention is recommended.")
elif predicted_class == 3:
    print("The ECG shows signs of Bradycardia — a slower than normal heart rate. Correlate with symptoms and evaluate further if persistent.")
elif predicted_class == 4:
    print("ST Elevation detected — may indicate acute myocardial infarction. Requires urgent evaluation.")
elif predicted_class == 5:
    print("Fusion beats detected — may be benign but monitor frequency and symptoms.")
else:
    print("Unrecognized pattern. Consider re-testing with better signal quality.")

# === Plot ECG signal ===
plt.figure(figsize=(12, 4))
plt.plot(plot_signal, color='blue', linewidth=1)
plt.title(f"🫀 ECG Signal — Predicted: {class_name} ({confidence:.2f}%)")
plt.xlabel("Sample Index")
plt.ylabel("Raw Voltage")
plt.grid(True)
plt.tight_layout()
plt.show()
