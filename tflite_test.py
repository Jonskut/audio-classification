import numpy as np
import librosa
from pathlib import Path
from tensorflow.keras.models import load_model

"""
Honda CB350 1971.WAV by yvesabox -- https://freesound.org/s/643605/ -- License: Attribution 4.0
Bike 250cc Start And Idle by A.Warner -- https://freesound.org/s/764546/ -- License: Attribution NonCommercial 4.0
09526 fast motorbike drive away.wav by Robinhood76 -- https://freesound.org/s/532172/ -- License: Attribution NonCommercial 4.0
"""

# Preprocess audio files
def preprocess_audio(file_path, sr=22050):
    """
    Preprocess a single audio file into a spectrogram, normalized and resized.
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # Generate the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize and reshape
    normalized_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)
    resized_spec = librosa.util.fix_length(normalized_spec, size=128, axis=1)
    return resized_spec[..., np.newaxis]  # Add channel dimension

# Perform inference on all files in a folder
def infer_from_folders(test_data_path, keras_model_path, sr=22050):
    """
    Perform inference on audio files in two subfolders ('car-sounds' and 'bike-sounds').
    """
    # Load the Keras model
    model = load_model(keras_model_path)

    # Subfolder paths for ground truth
    car_path = Path(test_data_path) / "car-test"
    bike_path = Path(test_data_path) / "bike-test"

    # Process and infer for each audio file
    results = []
    correct_predictions = 0
    total_predictions = 0

    # Helper function to process a folder
    def process_folder(folder_path, label):
        nonlocal correct_predictions, total_predictions
        for file in folder_path.glob("*.wav"):
            print(f"Processing file: {file}")
            
            # Preprocess the audio file
            preprocessed_audio = preprocess_audio(file, sr=sr)
            preprocessed_audio = np.expand_dims(preprocessed_audio, axis=0)  # Add batch dimension
            
            # Perform inference
            prediction_prob = model.predict(preprocessed_audio)
            prediction = "Car" if prediction_prob < 0.5 else "Motorcycle"
            
            # Compare prediction with ground truth
            is_correct = prediction == label
            correct_predictions += is_correct
            total_predictions += 1
            
            # Append results
            results.append({
                "file": file.name,
                "prediction": prediction,
                "ground_truth": label,
                "is_correct": is_correct,
                "confidence": prediction_prob[0][0],
            })
    
    # Process each folder
    process_folder(car_path, "Car")
    process_folder(bike_path, "Motorcycle")

    # Calculate accuracy
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correct)")
    else:
        print("\nNo predictions made.")
    
    return results

# Test data path and Keras model
test_data_path = "test-data"  # Path containing 'car-test' and 'bike-test' folders
keras_model_path = "fart-noises-v1.1.keras"  # Path to Keras model

# Perform inference
results = infer_from_folders(test_data_path, keras_model_path)

# Print results in tabular format
# Find the longest file name and label to calculate column widths
max_filename_length = max(len(result['file']) for result in results)
max_prediction_length = max(len(result['prediction']) for result in results)
max_ground_truth_length = max(len(result['ground_truth']) for result in results)

# Print header
print(f"{'File':<{max_filename_length}} {'Prediction':<{max_prediction_length}} {'Ground Truth':<{max_ground_truth_length}} {'Confidence':<10} {'Correct'}")
print("=" * (max_filename_length + max_prediction_length + max_ground_truth_length + 20))

# Print results
for result in results:
    is_correct = "✔" if result['is_correct'] else "✘"
    prediction_color = "\033[92m" if result['prediction'] == result['ground_truth'] else "\033[91m"
    confidence = result['confidence']
    print(f"{result['file']:<{max_filename_length}} {prediction_color}{result['prediction']:<{max_prediction_length}}\033[0m      {result['ground_truth']:<{max_ground_truth_length}} {confidence:<10.2f} {is_correct}")
