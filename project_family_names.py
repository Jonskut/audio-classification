"""
Jonatan Tevaniemi 150176680
Pelto Tian 150180676
Audio binary classification for car and motorcycle sounds
training data: https://freesound.org/people/Lauri_Lehtonen/#packs
"""

# Importing libraries for audio processing
import numpy as np
import matplotlib.pyplot as plt
import librosa

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import L2
from pathlib import Path

# Feature extraction
def extract_features(sounds, sr):
    features = []
    for sound in sounds:
        mel_spec = librosa.feature.melspectrogram(y=sound, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        print("Max value in log_mel_spec: ", np.max(log_mel_spec))
        padded_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=1)
        features.append(padded_spec)
    return np.array(features)

# Load by iterating over audio files in car-sounds and bike-sounds regardless of filename
car_sounds = []
car_path = Path('car-sounds')
for file in car_path.glob('*.wav'):
    y, sr = librosa.load(file, sr=22050)
    car_sounds.append(y)

bike_sounds = []
bike_path = Path('bike-sounds')
for file in bike_path.glob('*.wav'):
    y, sr = librosa.load(file, sr=22050)
    bike_sounds.append(y)

print("Files loaded")
print(len(car_sounds))
print(len(bike_sounds))

# Extract features and preprocess
car_features = extract_features(car_sounds, sr)
bike_features = extract_features(bike_sounds, sr)

print("length of car features: ", len(car_features))
print("length of bike features: ", len(bike_features))

# Combine features and labels
#x = np.array(car_features + bike_features)
X = np.concatenate((car_features, bike_features), axis=0)
y = np.array([0] * len(car_features) + [1] * len(bike_features))

print("shape of x: ", X.shape)
print("length of y: ", len(y))

# Reshape and normalize#
# X = X[..., np.newaxis] / 80
X = (X - np.mean(X)) / np.std(X)


print("range minimum of x: ", np.min(X))
print("range maximum of x: ", np.max(X))

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model
model.summary()

history = model.fit(
    x_train, y_train, 
    batch_size=3,
    validation_data=(x_test, y_test),
    epochs = 10,
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()