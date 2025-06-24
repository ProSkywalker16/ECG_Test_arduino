import wfdb
import numpy as np
import tensorflow as tf
from collections import Counter

# --- Config ---
record_id = '100'  # Change as needed
record_path = f'../mitbih_data/{record_id}'
target_symbols = ['N', 'A', 'V', 'L', 'R', 'f']

# --- Label maps ---
arrhythmia_map = {
    'N': 0,  # Normal
    'A': 1,  # Atrial Fibrillation
    'V': 2,  # Premature Ventricular Contraction
    'L': 3,  # Left Bundle Branch Block
    'R': 4,  # Right Bundle Branch Block
    'f': 5   # Fusion of ventricular and normal beat
}
decode_class = {v: k for k, v in arrhythmia_map.items()}
diagnosis_terms = {
    'N': 'Normal Sinus Rhythm (NSR)',
    'A': 'Atrial Fibrillation (AFib)',
    'V': 'Premature Ventricular Contraction (PVC)',
    'L': 'Left Bundle Branch Block (LBBB)',
    'R': 'Right Bundle Branch Block (RBBB)',
    'f': 'Fusion of ventricular and normal beat (Fusion Beat)'
}

# --- Load record and annotations ---
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="ecg_model_weighted.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Storage ---
results = []
confidences = []
timestamps = []

print(f"\n📊 ECG Prediction from MIT-BIH Record {record_id}")

# --- Predict beats for each arrhythmia type ---
for symbol in target_symbols:
    matches = [i for i, s in enumerate(annotation.symbol) if s == symbol]
    if not matches:
        continue
    print(f"\n--- 🔍 Found {len(matches)} annotation(s) for '{symbol}' ({diagnosis_terms[symbol]}) ---")

    for i in matches[:10]:  # Increase to 10 for better average
        sample_index = annotation.sample[i]
        start = max(0, sample_index - 900)
        end = start + 1800
        signal = record.p_signal[start:end, 0]

        if len(signal) < 1800:
            continue

        # Normalize
        signal = (signal - np.mean(signal)) / np.std(signal)
        input_data = signal.astype(np.float32).reshape(1, 1800, 1)

        # Predict
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        expected = arrhythmia_map[symbol]
        pred_symbol = decode_class[pred_class]

        print(f"🦠 Predicted: {pred_symbol} ({pred_class}) | Expected: {symbol} ({expected}) | Confidence: {confidence:.2f}% | Sample: {sample_index}")

        results.append(pred_symbol)
        confidences.append((pred_symbol, confidence))
        timestamps.append(sample_index)

# --- Medical Summary ---
print("\n--- 📋 Medical Interpretation Summary ---")
pred_counts = Counter(results)

for symbol, count in pred_counts.items():
    print(f"• {count} beat(s) classified as {diagnosis_terms[symbol]}")

# --- Medical Analysis ---
print("\n🧪 Medical Interpretation:")
if 'V' in pred_counts:
    print("The ECG record shows signs of Premature Ventricular Contractions (PVCs), indicating possible ventricular ectopy.")
if 'A' in pred_counts:
    print("The presence of Atrial Fibrillation (AFib) suggests the patient may have atrial arrhythmia.")
if 'L' in pred_counts or 'R' in pred_counts:
    print("Bundle branch blocks detected. May suggest conduction issues within the ventricles.")
if 'f' in pred_counts:
    print("Fusion beats identified, which are typically seen in ventricular arrhythmias.")
if not any(sym in pred_counts for sym in ['A', 'V', 'L', 'R', 'f']):
    print("The ECG shows predominantly Normal Sinus Rhythm (NSR), suggesting no apparent arrhythmia.")

# --- Emergency Detection ---
# --- Emergency Detection ---
print("\n🚨 Emergency Detection:")

afib_count = results.count('A')
pvc_count = results.count('V')
fusion_count = results.count('f')
lbbb_count = results.count('L')
rbbb_count = results.count('R')
nsr_count = results.count('N')
total_beats = len(results)

afib_confidences = [conf for sym, conf in confidences if sym == 'A']
avg_afib_conf = np.mean(afib_confidences) if afib_confidences else 0
afib_ratio = afib_count / total_beats if total_beats else 0
nsr_ratio = nsr_count / total_beats if total_beats else 0

emergency = False

# Rule 1: Co-occurrence AFib + PVC
if afib_count >= 10 and pvc_count >= 2 and avg_afib_conf >= 95:
    print("🆘 EMERGENCY: Co-occurrence of AFib and PVCs — atrial and ventricular instability.")
    emergency = True

# Rule 2: Frequent and confident AFib
elif afib_count >= 15 and avg_afib_conf >= 95 and afib_ratio > 0.3:
    print("🆘 EMERGENCY: Frequent Atrial Fibrillation detected.")
    emergency = True

# Rule 3: Frequent PVCs
elif pvc_count >= 5:
    print("🆘 EMERGENCY: Frequent Premature Ventricular Contractions detected.")
    emergency = True

# Rule 4: Fusion + PVC
elif fusion_count >= 1 and pvc_count >= 1:
    print("🆘 EMERGENCY: Fusion beats co-occurring with PVCs — possible ventricular arrhythmia.")
    emergency = True

# Rule 5: Bundle branch block
elif lbbb_count + rbbb_count >= 2:
    print("🆘 EMERGENCY: Multiple bundle branch blocks detected.")
    emergency = True

# Suppress if mostly NSR
elif nsr_ratio > 0.5:
    print("✅ Most beats are NSR. Emergency suppressed.")
    emergency = False

# Final fallback
if not emergency:
    print("✅ No emergency indicators detected.")
