# Import necessary libraries
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import parselmouth
from parselmouth.praat import call
import os

# Create an instance of the Flask application
app = Flask(__name__)

# --- Load Your Trained Models ---
# Load the scaler object
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the machine learning model
with open('parkinsons_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Define the Feature Extraction Function ---
# This is the complex function to extract the 22 features from an audio file
def extract_features(file_path):
    try:
        sound = parselmouth.Sound(file_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        
        # Fundamental frequency features
        mean_f0 = call(sound, "Get pitch", 0.0, 0.0, "Hertz", "parabolic", 0.0, 0.0)
        # Placeholder for actual mean F0 calculation, as Get Pitch returns an object.
        # This is a simplification. A more robust way would be to analyze the Pitch object.
        # For now, we'll use a simplified placeholder value from the Pitch object itself.
        mean_f0_val = call(sound.to_pitch(), "Get mean", 0, 0, "Hertz")

        f0_max = call(sound.to_pitch(), "Get maximum", 0, 0, "Hertz", "Parabolic")
        f0_min = call(sound.to_pitch(), "Get minimum", 0, 0, "Hertz", "Parabolic")

        # Jitter features
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100 # In %
        local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Shimmer features
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        local_db_shimmer = call([sound, point_process], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11_shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # HNR and NHR
        harmonics = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonics, "Get mean", 0, 0)
        nhr = 1 / (10**(hnr/10)) # Approximate NHR from HNR

        # The original dataset has 22 features excluding name and status.
        # Some features like RPDE, D2, DFA, spread1, spread2, PPE are complex and not directly available in parselmouth.
        # We will use placeholders (0) for them. For a real-world application, these would need specific algorithms.
        features = [
            mean_f0_val, f0_max, f0_min,
            local_jitter, local_absolute_jitter, rap_jitter, ppq5_jitter, ddp_jitter,
            local_shimmer, local_db_shimmer, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer,
            nhr, hnr,
            0, # RPDE placeholder
            0, # D2 placeholder
            0, # DFA placeholder
            0, # spread1 placeholder
            0, # spread2 placeholder
            0, # PPE placeholder
        ]
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# --- Define Flask Routes ---

# This is the route for your main homepage
@app.route('/')
def home():
    # It will serve the index.html file from your 'templates' folder
    return render_template('index.html')

# This is the route that will handle the audio processing
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an empty file part
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file temporarily
        # It's good practice to save it in a dedicated temporary folder
        temp_dir = 'temp_audio'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        # 1. Extract features from the saved audio file
        raw_features = extract_features(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)

        if raw_features is None:
            return jsonify({'error': 'Could not process the audio file. Please try a different recording.'}), 500
        
        # 2. Reshape features and scale them using the loaded scaler
        features_array = np.array(raw_features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        
        # 3. Make a prediction using the loaded model
        prediction = model.predict(scaled_features)
        
        # 4. Get a confidence score
        # decision_function gives a score where the sign indicates the class.
        # We can convert this to a pseudo-probability for the frontend.
        confidence_score = model.decision_function(scaled_features)[0]
        confidence = (1.0 / (1.0 + np.exp(-confidence_score))) # Sigmoid function

        # 5. Prepare the result in JSON format to send back to the frontend
        result = {
            'prediction': int(prediction[0]),
            'confidence': float(confidence)
        }
        
        return jsonify(result)

# This allows you to run the app directly
if __name__ == '__main__':
    app.run(debug=True)