import wfdb
import numpy as np
import tensorflow as tf
from collections import Counter

# Configuration
record_id = '118'  # You can change to test other records
target_symbols = ['A', 'V', 'L', 'R', 'f', 'N']
arrhythmia_map = {'N': 0, 'A': 1, 'V': 2, 'L': 3, 'R': 4, 'f': 5}
decode_class = {v: k for k, v in arrhythmia_map.items()}
description = {
    'N': "Normal Sinus Rhythm (NSR)",
    'A': "Atrial Fibrillation (AFib)",
    'V': "Premature Ventricular Contraction (PVC)",
    'L': "Left Bundle Branch Block (LBBB)",
    'R': "Right Bundle Branch Block (RBBB)",
    'f': "Fusion of Ventricular and Normal Beat"
}

# Load record
record_path = f"../mitbih_data/{record_id}"
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

# Load model
interpreter = tf.lite.Interpreter(model_path="ecg_model_weighted.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Collect predictions
results = []

for symbol in target_symbols:
    indices = [i for i, s in enumerate(annotation.symbol) if s == symbol]
    if not indices:
        continue

    print(f"\n--- 🔍 Found {len(indices)} annotation(s) for '{symbol}' ({description[symbol]}) ---")
    for idx in indices[:5]:  # Analyze up to 5 samples per class
        sample = annotation.sample[idx]
        start = max(0, sample - 900)
        end = start + 1800
        signal = record.p_signal[start:end, 0]
        if len(signal) != 1800:
            continue

        signal = (signal - np.mean(signal)) / np.std(signal)
        input_data = signal.astype(np.float32).reshape(1, 1800, 1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        pred_symbol = decode_class.get(pred_class, '?')

        print(f"🫀 Predicted: {pred_symbol} ({pred_class}) | Expected: {symbol} ({arrhythmia_map[symbol]})"
              f" | Confidence: {confidence:.2f}% | Sample: {sample}")

        results.append(pred_symbol)

# 📋 Summary
print("\n--- 📋 Medical Interpretation Summary ---")
summary = Counter(results)
for label, count in summary.items():
    print(f"• {count} beat(s) classified as {description[label]}")

# 🧠 Clinical Summary
summary_keys = set(summary.keys())
print("\n--- 🧠 Clinical Summary ---")

if summary_keys == {'N'}:
    print("The ECG record analyzed shows a Normal Sinus Rhythm (NSR) throughout. No signs of arrhythmia detected.")
elif 'A' in summary_keys and 'V' in summary_keys:
    print("The patient exhibits mostly normal heartbeats, but there are clear occurrences of both Atrial Fibrillation (AFib) "
          "and Premature Ventricular Contractions (PVCs), indicating possible atrial and ventricular arrhythmias. "
          "This mixed profile may reflect intermittent electrical instability, and clinical evaluation is advised. "
          "If symptoms such as dizziness, chest pain, or palpitations are present, urgent cardiology consultation is recommended.")
elif 'A' in summary_keys:
    print("The ECG indicates signs of Atrial Fibrillation (AFib), an arrhythmia originating in the atria. "
          "Clinical evaluation is recommended.")
elif 'V' in summary_keys and 'f' in summary_keys:
    print("The ECG shows Premature Ventricular Contractions (PVCs) along with fusion beats, suggesting mixed ventricular activity. "
          "This may indicate enhanced ventricular ectopy and warrants clinical monitoring.")
elif 'V' in summary_keys:
    print("The ECG indicates Premature Ventricular Contractions (PVCs), suggesting ventricular ectopy. "
          "If frequent, further evaluation is needed.")
elif 'L' in summary_keys or 'R' in summary_keys:
    print("Bundle branch blocks detected. May indicate conduction delays in the heart's electrical system. "
          "Needs correlation with patient symptoms.")
elif 'f' in summary_keys:
    print("Fusion beats detected. Can be benign but may occur alongside PVCs. Monitor for frequency and symptom correlation.")
else:
    print("Mixed or unclassified results. Consider re-evaluation with more samples or clinical correlation.")
