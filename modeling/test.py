import pandas as pd
import numpy as np
import joblib  # For saving and loading the trained models

# Function to verify the prediction on a sample
def verify_sample_prediction(model_filename, sample_filename):
    # Load the model
    model = joblib.load(model_filename)

    # Load the sample data
    sample_df = pd.read_csv(sample_filename)
    sample_features = sample_df.drop('DIAGNOSIS', axis=1)
    sample_label = sample_df['DIAGNOSIS'].values[0]

    # Predict using the model
    sample_pred = model.predict(sample_features.values.reshape(1, -1))[0]

    print(f"Model Prediction: {sample_pred}")
    print(f"True Label: {sample_label}")

    if sample_pred == sample_label:
        print("The model predicts correctly on this sample.\n")
    else:
        print("The model does NOT predict correctly on this sample.\n")

# Verify CN sample
verify_sample_prediction('saved_models/rf_mmse.pkl', 'sample_CN_mmse.csv')

# Verify MCI sample
verify_sample_prediction('saved_models/rf_mmse_moca.pkl', 'sample_MCI_mmse_moca.csv')

# Verify AD sample
verify_sample_prediction('saved_models/rf_mmse_moca_npiq.pkl', 'sample_AD_mmse_moca_npiq.csv')
