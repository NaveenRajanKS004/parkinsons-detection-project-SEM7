from flask import Flask
import pickle

# Create an instance of the Flask class
app = Flask(__name__)

# --- MODEL AND SCALER LOADING ---
# Load the trained model
model = pickle.load(open('parkinsons_model.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
# ------------------------------------

# Define a "route" for the home page
@app.route('/')
def home():
    # A simple message to confirm the server is running and model is loaded
    return 'Server is running. Model and scaler have been loaded.'

# This block runs the app when you execute the python file
if __name__ == '__main__':
    app.run(debug=True)