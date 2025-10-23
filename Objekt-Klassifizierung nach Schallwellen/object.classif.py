import tkinter as tk
from tkinter import filedialog
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import datetime
import csv
import os

# --------------------------
# CNN-Modell
# --------------------------
def build_model(input_shape=(128, 128, 1), num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# --------------------------
# Statistiken initialisieren
# --------------------------
stats_file = "vehicle_stats.csv"
if not os.path.exists(stats_file):
    with open(stats_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "VehicleClass"])

vehicle_counts = [0]*10  # 10 Klassen

# --------------------------
# Funktionen
# --------------------------
def generate_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Resize für CNN
    img = Image.fromarray(S_dB)
    img = img.resize((128,128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    img_array = np.expand_dims(img_array, axis=-1) # Kanaldimension für CNN
    return img_array

def log_vehicle_class(vehicle_class):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(stats_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, vehicle_class])
    vehicle_counts[vehicle_class] += 1
    update_dashboard()

def update_dashboard():
    text = "Fahrzeugstatistik:\n"
    for i, count in enumerate(vehicle_counts):
        text += f"Klasse {i}: {count}\n"
    stats_label.config(text=text)

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        spectrogram = generate_spectrogram(file_path)
        # Dummy-Vorhersage (Modell ist noch untrainiert)
        prediction = model.predict(spectrogram)
        predicted_class = np.argmax(prediction)
        result_label.config(text=f"Vorhergesagte Klasse: {predicted_class}")
        log_vehicle_class(predicted_class)

# --------------------------
# UI
# --------------------------
root = tk.Tk()
root.title("Live Fahrzeug-Klassifikation (10 Klassen)")

btn_select = tk.Button(root, text="Audiodatei auswählen", command=select_file)
btn_select.pack(pady=10)

result_label = tk.Label(root, text="Vorhergesagte Klasse: ")
result_label.pack(pady=5)

stats_label = tk.Label(root, text="Fahrzeugstatistik:\n" + "\n".join([f"Klasse {i}: 0" for i in range(10)]))
stats_label.pack(pady=10)

root.mainloop()
