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

#CNN 10 Klassen
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

# Funktionen
def generate_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(6,4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spektrogramm')
    plt.show()

    # Resize für CNN
    img = Image.fromarray(S_dB)
    img = img.resize((128,128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        spectrogram = generate_spectrogram(file_path)
        # Dummy-Vorhersage (Modell ist noch untrainiert)
        prediction = model.predict(spectrogram)
        predicted_class = np.argmax(prediction)
        result_label.config(text=f"Vorhergesagte Klasse: {predicted_class}")

# UI
root = tk.Tk()
root.title("Audio Klassifikation (10 Klassen)")

btn_select = tk.Button(root, text="Audiodatei auswählen", command=select_file)
btn_select.pack(pady=20)

result_label = tk.Label(root, text="Vorhergesagte Klasse: ")
result_label.pack(pady=10)

root.mainloop()
