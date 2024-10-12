import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the NPIQ model
model_filename = 'saved_models/lr_npiq.pkl'  # Adjust the filename if necessary
try:
    model, feature_names = joblib.load(model_filename)
    print("Model and feature names loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found: {model_filename}")
    exit()

# Create a test input with all zeros
def create_all_zero_test_data(feature_names):
    test_data = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    return test_data

# Generate the all-zero test data
test_data = create_all_zero_test_data(feature_names)

# Display the test data
print("\nTest Data (All Zeros):")
print(test_data)

# Make predictions
predictions = model.predict(test_data)

# Since we don't have a true label for this synthetic data, we'll just display the prediction
print("\nPrediction for All-Zero Input:")
diagnosis_map = {
    0: 'Normal',
    1: 'Mild Cognitive Impairment',
    2: 'Alzheimer\'s Disease'
}
predicted_diagnosis = diagnosis_map.get(predictions[0], 'Unknown')
print(f"Predicted Diagnosis: {predicted_diagnosis}")
