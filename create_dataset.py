import os
import pandas as pd
import parselmouth
from parselmouth.praat import call
import numpy as np

# This is the exact same function from our app.py
def extract_features(file_path):
    try:
        sound = parselmouth.Sound(file_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 60, 600)
        mean_f0_val = call(sound.to_pitch(), "Get mean", 0, 0, "Hertz")
        f0_max = call(sound.to_pitch(), "Get maximum", 0, 0, "Hertz", "Parabolic")
        f0_min = call(sound.to_pitch(), "Get minimum", 0, 0, "Hertz", "Parabolic")
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        local_db_shimmer = local_shimmer
        apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11_shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonics = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonics, "Get mean", 0, 0)
        nhr = 1 / (10**(hnr/10))
        return [
            mean_f0_val, f0_max, f0_min, local_jitter, local_absolute_jitter,
            rap_jitter, ppq5_jitter, ddp_jitter, local_shimmer, local_db_shimmer,
            apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer, nhr, hnr
        ]
    except Exception as e:
        print(f"Could not process {os.path.basename(file_path)}: {e}")
        return None

# The names of the features we are extracting
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 
    'Shimmer:DDA', 'NHR', 'HNR'
]

# Process all audio files and build our dataset
data_rows = []
# Process healthy controls (status=0)
for filename in os.listdir('HC_AH'):
    if filename.endswith('.wav'):
        filepath = os.path.join('HC_AH', filename)
        features = extract_features(filepath)
        if features:
            data_rows.append(features + [0]) # Add features and status label 0

# Process Parkinson's patients (status=1)
for filename in os.listdir('PD_AH'):
    if filename.endswith('.wav'):
        filepath = os.path.join('PD_AH', filename)
        features = extract_features(filepath)
        if features:
            data_rows.append(features + [1]) # Add features and status label 1

# Create a DataFrame and save it to a new CSV file
column_names = feature_names + ['status']
df = pd.DataFrame(data_rows, columns=column_names)
df.to_csv('tremorsense_dataset.csv', index=False)

print(f"--- Success! ---")
print(f"Created new dataset 'tremorsense_dataset.csv' with {len(df)} samples.")