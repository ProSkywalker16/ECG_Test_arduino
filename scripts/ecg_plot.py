import csv
import os
import threading
import time
from collections import deque
import tkinter as tk
from tkinter import messagebox, filedialog

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import serial
import serial.tools.list_ports

# ----- Configuration -----
BAUDRATE          = 19200
PLOT_WINDOW       = 5       # seconds to display in the rolling plot
SAMPLE_RATE       = 60      # nominal samples per second
ADC_MAX           = 1023
VREF              = 5.0     # volts
CAPTURE_DURATION  = 40      # auto‐stop after this many seconds
# --------------------------

def next_filename(base="ecg_plot", ext=".csv"):
    """Find the next unused filename like ecg_plot1.csv, ecg_plot2.csv, ..."""
    idx = 1
    while True:
        name = f"{base}{idx}{ext}"
        if not os.path.exists(name):
            return name
        idx += 1

class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Monitor")

        # Data buffers
        self.raw_data = []  # full session: (t, raw, voltage)
        self.buffer   = deque(maxlen=int(PLOT_WINDOW * SAMPLE_RATE))

        # Serial port & state
        self.ser = None
        self.running = False
        self.start_time = None

        # UI buttons
        self.start_btn  = tk.Button(root, text="Start",  command=self.start_capture)
        self.stop_btn   = tk.Button(root, text="Stop",   state=tk.DISABLED, command=self.stop_capture)
        self.export_btn = tk.Button(root, text="Export CSV", state=tk.DISABLED, command=self.export_csv)
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_btn.pack(side=tk.LEFT,  padx=5, pady=5)
        self.export_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Plot setup
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.line, = self.ax.plot([], [], lw=1)
        self.ax.set_xlim(0, PLOT_WINDOW)
        self.ax.set_ylim(0, VREF)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.set_title("Real-time ECG")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ani = None

        # Auto-start capture
        self.start_capture()

    def find_arduino(self):
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if "arduino" in p.description.lower():
                return p.device
        if ports:
            return ports[0].device
        raise IOError("No serial ports found")

    def start_capture(self):
        try:
            port = self.find_arduino()
            self.ser = serial.Serial(port, BAUDRATE, timeout=1)
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            return

        # reset buffers and UI
        self.raw_data.clear()
        self.buffer.clear()
        self.start_time = time.time()
        self.running = True

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.DISABLED)

        # thread to read serial
        threading.Thread(target=self._read_serial, daemon=True).start()

        # animation
        self.ani = animation.FuncAnimation(self.fig, self._update_plot, interval=50, blit=False)
        self.canvas.draw()

        # schedule auto-stop
        self.root.after(int(CAPTURE_DURATION * 1000), self.stop_capture)

    def stop_capture(self):
        if not self.running:
            return
        self.running = False
        time.sleep(0.1)
        if self.ser and self.ser.is_open:
            self.ser.close()

        # auto-save
        fname = next_filename()
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s','raw_adc','voltage_V'])
            writer.writerows(self.raw_data)

        messagebox.showinfo("Capture complete", f"Data auto-saved as:\n{fname}")

        # update button states
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.NORMAL)

    def export_csv(self):
        path = tk.filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv")]
        )
        if not path:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s','raw_adc','voltage_V'])
            writer.writerows(self.raw_data)
        messagebox.showinfo("Export Complete", f"Data saved as:\n{path}")

    def _read_serial(self):
        while self.running:
            line = self.ser.readline().decode('ascii', errors='ignore').strip()
            if not line or line.startswith("ECG"):
                continue
            parts = line.split(',')
            try:
                raw = int(parts[0])
            except Exception:
                continue

            voltage = raw * (VREF / ADC_MAX)
            elapsed = time.time() - self.start_time

            # store data
            self.raw_data.append((elapsed, raw, voltage))
            self.buffer.append((elapsed, voltage))

    def _update_plot(self, _):
        if not self.buffer:
            return
        times, volts = zip(*self.buffer)
        t_end = times[-1]
        t_start = max(0, t_end - PLOT_WINDOW)
        self.line.set_data(times, volts)
        self.ax.set_xlim(t_start, t_start + PLOT_WINDOW)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()