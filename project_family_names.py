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
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Feature extraction
def extract_features(sounds, sr):
    features = []
    for sound in sounds:
        mel_spec = librosa.feature.melspectrogram(y=sound, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        padded_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=1)
        features.append(padded_spec)
    return np.array(features)

# Data augmentation for audio
# def augment_audio(y, sr):
#     augmented = []
#     # Time stretch
#     augmented.append(librosa.effects.time_stretch(y, rate=1.1))
#     augmented.append(librosa.effects.time_stretch(y, rate=0.9))
#     # Pitch shift
#     augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
#     augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
#     # Add Gaussian noise
#     augmented.append(y + 0.005 * np.random.randn(len(y)))
#     return augmented

# Load and augment data
car_sounds = []
car_path = Path('car-sounds')
for file in car_path.glob('*.wav'):
    y, sr = librosa.load(file, sr=22050)
    car_sounds.append(y)
    #car_sounds.extend(augment_audio(y, sr))  # Augment car sounds

bike_sounds = []
bike_path = Path('bike-sounds')
for file in bike_path.glob('*.wav'):
    y, sr = librosa.load(file, sr=22050)
    bike_sounds.append(y)
    #bike_sounds.extend(augment_audio(y, sr))  # Augment bike sounds

print("Files loaded and augmented")
print("Number of car sounds:", len(car_sounds))
print("Number of bike sounds:", len(bike_sounds))

# Extract features and preprocess
car_features = extract_features(car_sounds, sr)
bike_features = extract_features(bike_sounds, sr)

print("Number of car features:", len(car_features))
print("Number of bike features:", len(bike_features))

# Combine features and labels
X = np.concatenate((car_features, bike_features), axis=0)
y = np.array([0] * len(car_features) + [1] * len(bike_features))

print("Shape of X:", X.shape)
print("Length of y:", len(y))

# Normalize data
X = (X - np.mean(X)) / np.std(X)

# Reshape for CNN
X = X[..., np.newaxis]

print("Range of X after normalization: min =", np.min(X), ", max =", np.max(X))

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor = 0.8,
    patience= 3,
    min_lr=1e-5,
    verbose=1,
)

# Define the model
model = Sequential([
    Conv2D(10, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(20, (3, 3), activation='relu', kernel_regularizer=L2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    GlobalAveragePooling2D(),

    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=7,
    validation_data=(x_val, y_val),
    epochs=100,
    callbacks=[reduce_lr]
)

# Plot training and validation performance
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.title('Model Performance')
plt.show()

model.save("fart-noises-v1.1.keras")
