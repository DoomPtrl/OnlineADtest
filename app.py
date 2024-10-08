from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os
from questionnaire_data import questionnaires

app = Flask(__name__)

# Diagnosis mapping (replace with your actual diagnosis labels)
diagnosis_map = {
    0: 'Normal',
    1: 'Mild Cognitive Impairment',
    2: 'Alzheimer\'s Disease'
}

# Build a mapping from feature to its question and range
feature_info_map = {}
for q in questionnaires:
    criteria = questionnaires[q]['criteria']
    ranges = questionnaires[q]['criteria_range']
    questions = questionnaires[q]['questions']
    for feature, range_, question in zip(criteria, ranges, questions):
        feature_info_map[feature] = {
            'question': question,
            'range': range_
        }

# Helper function to load the correct model
def load_model(selected_questionnaires):
    model_filename = 'saved_models/' + 'rf_' + '_'.join(sorted(selected_questionnaires)) + '.pkl'
    if os.path.exists(model_filename):
        model, feature_names = joblib.load(model_filename)
        return model, feature_names
    else:
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_questionnaires = request.form.getlist('questionnaires')
        if not selected_questionnaires:
            error = 'Please select at least one questionnaire.'
            return render_template('index.html', questionnaires=questionnaires, error=error)
        else:
            if 'mmse' in selected_questionnaires:
                return redirect(url_for('mmse'))
            elif 'moca' in selected_questionnaires:
                return redirect(url_for('moca'))
            elif 'npiq' in selected_questionnaires:
                return redirect(url_for('npiq'))
    else:
        return render_template('index.html', questionnaires=questionnaires)

@app.route('/mmse', methods=['GET', 'POST'])
def mmse():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        print("MMSE Form Data:", form_data)  # Debugging print statement
        selected_questionnaires = ['mmse']
        model, feature_names = load_model(selected_questionnaires)
        if model is None:
            error = 'Model not found for MMSE.'
            return render_template('mmse_input.html', error=error)
        
        # Collecting and ordering features
        features = []
        for feature in feature_names:
            value = form_data.get(feature, '0')  # Default value to '0' if not provided
            try:
                features.append(float(value))
            except ValueError:
                error = f"Invalid input for {feature}. Please enter a valid number."
                return render_template('mmse_input.html', error=error)
        
        features = np.array(features).reshape(1, -1)

        # Apply preprocessing if applicable
        scaler_filename = 'saved_models/scaler_mmse.pkl'
        if os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
            features = scaler.transform(features)
        
        # Make the prediction
        print(f"Features for prediction: {features}")  # Debugging print statement
        prediction = model.predict(features)
        predicted_diagnosis = diagnosis_map.get(prediction[0], 'Unknown')
        return render_template('result.html', prediction=predicted_diagnosis)

    return render_template('mmse_input.html')

@app.route('/moca', methods=['GET', 'POST'])
def moca():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        print("MOCA Form Data:", form_data)  # Debugging print statement
        selected_questionnaires = ['moca']
        model, feature_names = load_model(selected_questionnaires)
        if model is None:
            error = 'Model not found for MOCA.'
            return render_template('moca_input.html', error=error)
        
        # Collecting and ordering features
        features = []
        for feature in feature_names:
            value = form_data.get(feature, '0')  # Default value to '0' if not provided
            try:
                features.append(float(value))
            except ValueError:
                error = f"Invalid input for {feature}. Please enter a valid number."
                return render_template('moca_input.html', error=error)
        
        features = np.array(features).reshape(1, -1)

        # Apply preprocessing if applicable
        scaler_filename = 'saved_models/scaler_moca.pkl'
        if os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
            features = scaler.transform(features)
        
        # Make the prediction
        print(f"Features for prediction: {features}")  # Debugging print statement
        prediction = model.predict(features)
        predicted_diagnosis = diagnosis_map.get(prediction[0], 'Unknown')
        return render_template('result.html', prediction=predicted_diagnosis)

    return render_template('moca_input.html')

@app.route('/npiq', methods=['GET', 'POST'])
def npiq():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        print("NPIQ Form Data:", form_data)  # Debugging print statement
        selected_questionnaires = ['npiq']
        model, feature_names = load_model(selected_questionnaires)
        if model is None:
            error = 'Model not found for NPIQ.'
            return render_template('npiq_input.html', error=error)
        
        # Collecting and ordering features
        features = []
        for feature in feature_names:
            value = form_data.get(feature, '0')  # Default value to '0' if not provided
            try:
                features.append(float(value))
            except ValueError:
                error = f"Invalid input for {feature}. Please enter a valid number."
                return render_template('npiq_input.html', error=error)
        
        features = np.array(features).reshape(1, -1)

        # Apply preprocessing if applicable
        scaler_filename = 'saved_models/scaler_npiq.pkl'
        if os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
            features = scaler.transform(features)
        
        # Make the prediction
        print(f"Features for prediction: {features}")  # Debugging print statement
        prediction = model.predict(features)
        predicted_diagnosis = diagnosis_map.get(prediction[0], 'Unknown')
        return render_template('result.html', prediction=predicted_diagnosis)

    return render_template('npiq_input.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)