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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from pathlib import Path

# Load by iterating over audio files in car-sounds and bike-sounds regardless of filename
car_sounds = []
car_path = Path('car-sounds')
for file in car_path.glob('*.wav'):
    y, sr = librosa.load(file)
    car_sounds.append(y)

bike_sounds = []
bike_path = Path('bike-sounds')
for file in bike_path.glob('*.wav'):
    y, sr = librosa.load(file)
    bike_sounds.append(y)



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
# Extract features and preprocess
# car_features = []
# for sound in car_sounds:
#     mel_spec = librosa.feature.melspectrogram(y=sound, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
#     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
#     resized_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=1)
#     car_features.append(resized_spec)

# bike_features = []
# for sound in bike_sounds:
#     mel_spec = librosa.feature.melspectrogram(y=sound, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
#     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
#     resized_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=1)
#     bike_features.append(resized_spec)

# # Combine features and labels
# X = np.array(car_features + bike_features)
# y = np.array([0] * len(car_features) + [1] * len(bike_features))

# # Reshape and normalize
# X = X[..., np.newaxis] / np.max(X)

# # Split data
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
