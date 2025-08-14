import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

print("--- Retraining on our own custom dataset ---")
# UPDATED: Read from our newly created dataset
df = pd.read_csv('tremorsense_dataset.csv')
print("Successfully loaded tremorsense_dataset.csv")

# We use the same feature names, but now they are from our own data
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 
    'Shimmer:DDA', 'NHR', 'HNR'
]
X = df[features]
y = df['status']

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)
print("Final Random Forest model has been trained.")

with open('parkinsons_model_simple.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n--- Success! ---")
print("Definitive model 'parkinsons_model_simple.pkl' has been saved.")