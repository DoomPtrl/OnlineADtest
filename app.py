from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import os
from questionnaire_data import questionnaires
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure random key

# Diagnosis mapping
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
    model_filename = 'saved_models/' + 'lr_' + '_'.join(sorted(selected_questionnaires)) + '.pkl'
    if os.path.exists(model_filename):
        model, feature_names = joblib.load(model_filename)
        return model, feature_names
    else:
        return None, None

@app.route('/')
def home():
    # Get the list of completed questionnaires from the session
    completed_questionnaires = session.get('completed_questionnaires', [])
    return render_template('home.html', questionnaires=questionnaires, completed_questionnaires=completed_questionnaires)

@app.route('/<questionnaire>', methods=['GET', 'POST'])
def questionnaire_route(questionnaire):
    if questionnaire not in questionnaires:
        return "Questionnaire not found.", 404

    template_name = f"{questionnaire}_input.html"

    if request.method == 'POST':
        form_data = {key: float(value) if value.replace('.', '', 1).isdigit() else value for key, value in request.form.to_dict().items()}
        print(f"{questionnaire.upper()} Form Data:", form_data)  # Debugging print statement

        # Store the form data in session
        if 'questionnaire_data' not in session:
            session['questionnaire_data'] = {}
        session['questionnaire_data'][questionnaire] = form_data

        # Update the list of completed questionnaires
        completed_questionnaires = session.get('completed_questionnaires', [])
        if questionnaire not in completed_questionnaires:
            completed_questionnaires.append(questionnaire)
            session['completed_questionnaires'] = completed_questionnaires

        # Redirect back to the home page
        return redirect(url_for('home'))

    # GET request: Load the questionnaire with previous answers if available
    # Retrieve previous answers from the session
    previous_answers = {}
    questionnaire_data = session.get('questionnaire_data', {})
    if questionnaire in questionnaire_data:
        previous_answers = questionnaire_data[questionnaire]

    return render_template(template_name, previous_answers=previous_answers)

@app.route('/predict')
def predict():
    # Get the list of completed questionnaires
    completed_questionnaires = session.get('completed_questionnaires', [])
    if not completed_questionnaires:
        error = 'Please complete at least one questionnaire before predicting.'
        return render_template('home.html', questionnaires=questionnaires, completed_questionnaires=[], error=error)

    # Load the appropriate model
    model, feature_names = load_model(completed_questionnaires)
    if model is None:
        error = 'Model not found for the selected questionnaires.'
        return render_template('home.html', questionnaires=questionnaires, completed_questionnaires=completed_questionnaires, error=error)

    # Collect the form data
    questionnaire_data = session.get('questionnaire_data', {})
    features = []
    for feature in feature_names:
        value = None
        # Find the questionnaire that contains this feature
        for q in completed_questionnaires:
            if feature in questionnaires[q]['criteria']:
                value = questionnaire_data[q].get(feature, 0)
                break
        if value is None:
            value = '0'  # Default to 0 if not found
        try:
            features.append(float(value) if isinstance(value, (int, float)) or value.replace('.', '', 1).isdigit() else 0.0)
        except ValueError:
            error = f"Invalid input for {feature}. Please enter a valid number."
            return render_template('home.html', error=error)

    # Create a DataFrame with the features and feature names
    features_df = pd.DataFrame([features], columns=feature_names)

    # Make the prediction
    print(f"Features for prediction:\n{features_df}")  # Debugging print statement
    prediction = model.predict(features_df)
    predicted_diagnosis = diagnosis_map.get(prediction[0], 'Unknown')

    # Clear the session data if you want to reset after prediction
    # session.clear()

    return render_template('result.html', prediction=predicted_diagnosis)

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
