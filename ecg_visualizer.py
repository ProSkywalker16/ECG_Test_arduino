import serial
import matplotlib.pyplot as plt
from collections import deque
import time
import csv
import os

# Configuration
PORT = 'COM11'       # Change to your Arduino's port
BAUD_RATE = 9600
MAX_POINTS = 1000    # Number of points to display
SAVE_TO_CSV = True   # Set to True to save all data
CAPTURE_DURATION = 60  # Stop capturing after 30 seconds

# Ensure the /raw directory exists
RAW_DIR = "D:/ECG_Test_arduino/data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Function to find the next available filename
def get_next_filename(base_name):
    counter = 0
    while True:
        filename = f"{base_name}_{counter}.csv" if counter else f"{base_name}.csv"
        full_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

# Get the next available file name
csv_filepath = get_next_filename("ecg_data")

# Initialize plot
plt.ion()  # Interactive mode
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], 'b-')
ax.set_title('Real-Time ECG Monitoring')
ax.set_xlabel('Time (s)')
ax.set_ylabel('ECG Value')
ax.grid(True)

# Data buffers
timestamps = deque(maxlen=MAX_POINTS)
ecg_values = deque(maxlen=MAX_POINTS)
start_time = time.time()
capture_active = True

# CSV file setup
if SAVE_TO_CSV:
    csv_file = open(csv_filepath, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'ECG Value', 'LO+ Status', 'LO- Status'])

try:
    # Initialize serial connection with error handling
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        ser.flushInput()
    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
        print("Check: 1) Correct COM port 2) Arduino is connected 3) No other program is using the port")
        capture_active = False

    while True:
        if capture_active and (time.time() - start_time >= CAPTURE_DURATION):
            print(f"{CAPTURE_DURATION} seconds elapsed. Stopping data capture.")
            capture_active = False
            ser.close()
            if SAVE_TO_CSV:
                csv_file.close()

        if capture_active and ser.in_waiting:
            try:
                raw_data = ser.readline()
                try:
                    line_data = raw_data.decode('utf-8').strip()
                except UnicodeDecodeError:
                    line_data = raw_data.decode('ascii', errors='ignore').strip()
                
                if line_data:
                    parts = line_data.split(',')
                    if len(parts) >= 3:
                        try:
                            ecg = float(parts[0])
                            lo_plus = float(parts[1])
                            lo_minus = float(parts[2])
                            
                            current_time = time.time() - start_time
                            
                            timestamps.append(current_time)
                            ecg_values.append(ecg)
                            
                            if SAVE_TO_CSV:
                                csv_writer.writerow([current_time, ecg, lo_plus, lo_minus])
                            
                            line.set_xdata(timestamps)
                            line.set_ydata(ecg_values)
                            ax.relim()
                            ax.autoscale_view()
                            
                        except (ValueError, IndexError) as e:
                            print(f"Data parsing error: {e}")
                            continue
            except Exception as e:
                print(f"Serial read error: {e}")
                continue

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Stopping visualization...")
finally:
    if 'ser' in locals() and capture_active:
        ser.close()
    if SAVE_TO_CSV and 'csv_file' in locals():
        csv_file.close()
    print(f"Data capture complete. File saved to {csv_filepath}")
    plt.ioff()
    plt.show()
