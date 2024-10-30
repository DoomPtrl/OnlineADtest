import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import joblib
import json
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from flask import make_response

# Configure Logging with Rotation
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=3)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
handler.setFormatter(formatter)

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure random key
app.logger.addHandler(handler)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load questionnaires info from JSON file
with open('questionnaires_info.json', 'r') as f:
    questionnaires = json.load(f)

# Diagnosis mapping
diagnosis_map = {
    0: 'Normal',
    1: 'Mild Cognitive Impairment',
    2: "Alzheimer's Disease"
}

# Helper function to load the correct model
def load_model(selected_questionnaires):
    model_filename = 'saved_models/' + 'rf_' + '_'.join(sorted(selected_questionnaires)) + '.pkl'
    if os.path.exists(model_filename):
        try:
            model, feature_names = joblib.load(model_filename)
            app.logger.debug(f"Loaded model {model_filename} with features: {feature_names}")
            return model, feature_names
        except Exception as e:
            app.logger.error(f"Error loading model {model_filename}: {e}")
            return None, None
    else:
        app.logger.error(f"Model file {model_filename} not found.")
        return None, None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Define the nocache decorator
def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        resp = make_response(view(*args, **kwargs))
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    return no_cache

@app.route('/')
def home():
    session.clear()  # Clear the session data when the user goes back to the home page
    # Assuming patient data is stored in 'patient_info.csv'
    if os.path.exists('patient_info.csv'):
        patient_data = pd.read_csv('patient_info.csv', dtype=str)  # Read all columns as strings
    else:
        patient_data = pd.DataFrame(columns=['id', 'name', 'sex', 'age', 'address', 'photo', 'diagnosis'])
    # Convert DataFrame to a list of dictionaries for easier template rendering
    patient_list = patient_data.to_dict(orient='records')
    app.logger.debug(f"Patient List: {patient_list}")
    return render_template('patient_info.html', patients=patient_list)

@app.route('/add_patient', methods=['POST'])
def add_patient():
    # Clear the session data when a new patient is added
    session.clear()
    # Get form data
    new_patient_data = request.form.to_dict()
    if os.path.exists('patient_info.csv'):
        patient_data = pd.read_csv('patient_info.csv', dtype=str)  # Read all columns as strings
    else:
        patient_data = pd.DataFrame(columns=['id', 'name', 'sex', 'age', 'address', 'photo', 'diagnosis'])
    if not patient_data.empty:
        new_id = int(patient_data['id'].max()) + 1
        new_patient_data['id'] = str(new_id)
    else:
        new_patient_data['id'] = '1'
    try:
        new_patient_data['age'] = str(int(new_patient_data['age']))  # Convert age to string
    except ValueError:
        new_patient_data['age'] = '0'  # Default age if invalid input
        app.logger.warning(f"Invalid age input for new patient: {new_patient_data}")

    new_patient_data['diagnosis'] = 'Unknown'

    # Handle file upload
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo and allowed_file(photo.filename):
            filename = secure_filename(photo.filename)
            # Rename the file to include patient ID
            filename = f"patient_{new_patient_data['id']}_{filename}"
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_patient_data['photo'] = filename
            app.logger.debug(f"Uploaded photo for patient {new_patient_data['id']}: {filename}")
        else:
            new_patient_data['photo'] = 'default.jpg'  # Use a default image
            app.logger.debug(f"No valid photo uploaded for patient {new_patient_data['id']}. Using default.")
    else:
        new_patient_data['photo'] = 'default.jpg'
        app.logger.debug(f"No photo field in form for patient {new_patient_data['id']}. Using default.")

    # Save the new patient data
    new_patient_df = pd.DataFrame([new_patient_data])
    patient_data = pd.concat([patient_data, new_patient_df], ignore_index=True)
    patient_data.to_csv('patient_info.csv', index=False)

    app.logger.debug(f"Added new patient: {new_patient_data}")

    return redirect(url_for('home'))

@app.route('/edit_patient/<int:patient_id>', methods=['GET', 'POST'])
def edit_patient(patient_id):
    if os.path.exists('patient_info.csv'):
        patient_data = pd.read_csv('patient_info.csv', dtype=str)  # Read all columns as strings
    else:
        patient_data = pd.DataFrame(columns=['id', 'name', 'sex', 'age', 'address', 'photo', 'diagnosis'])
    if request.method == 'POST':
        # Update patient information
        updated_patient_data = request.form.to_dict()
        updated_patient_data['id'] = str(patient_id)
        try:
            updated_patient_data['age'] = str(int(updated_patient_data['age']))  # Convert age to string
        except ValueError:
            updated_patient_data['age'] = '0'  # Default age if invalid input
            app.logger.warning(f"Invalid age input for patient {patient_id}: {updated_patient_data}")

        # Handle file upload
        if 'photo' in request.files and request.files['photo'].filename != '':
            photo = request.files['photo']
            if photo and allowed_file(photo.filename):
                filename = secure_filename(photo.filename)
                filename = f"patient_{patient_id}_{filename}"
                photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                updated_patient_data['photo'] = filename
                app.logger.debug(f"Uploaded new photo for patient {patient_id}: {filename}")
            else:
                # Keep the existing photo if upload fails
                existing_photo = patient_data.loc[patient_data['id'] == str(patient_id), 'photo'].values[0]
                updated_patient_data['photo'] = existing_photo
                app.logger.warning(f"Invalid photo upload for patient {patient_id}. Keeping existing photo.")
        else:
            # Keep the existing photo if no new photo is uploaded
            existing_photo = patient_data.loc[patient_data['id'] == str(patient_id), 'photo'].values[0]
            updated_patient_data['photo'] = existing_photo
            app.logger.debug(f"No new photo uploaded for patient {patient_id}. Keeping existing photo.")

        # Update the patient data
        patient_data.loc[patient_data['id'] == str(patient_id), ['name', 'sex', 'age', 'address', 'photo']] = [
            updated_patient_data['name'],
            updated_patient_data['sex'],
            updated_patient_data['age'],
            updated_patient_data['address'],
            updated_patient_data['photo']
        ]
        patient_data.to_csv('patient_info.csv', index=False)
        app.logger.debug(f"Edited patient {patient_id}: {updated_patient_data}")
        return redirect(url_for('home'))
    else:
        # Render the edit form with current patient data
        patient = patient_data.loc[patient_data['id'] == str(patient_id)].to_dict(orient='records')
        if not patient:
            app.logger.error(f"Attempted to edit non-existent patient ID: {patient_id}")
            return "Patient not found.", 404
        patient = patient[0]
        app.logger.debug(f"Editing patient: {patient}")
        return render_template('edit_patient.html', patient=patient)

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    if os.path.exists('patient_info.csv'):
        patient_data = pd.read_csv('patient_info.csv', dtype=str)  # Read all columns as strings
        # Remove the patient with the given ID
        patient_data = patient_data[patient_data['id'] != str(patient_id)]
        # Save the updated data
        patient_data.to_csv('patient_info.csv', index=False)
        app.logger.debug(f"Deleted patient with ID: {patient_id}")
    else:
        app.logger.warning(f"Attempted to delete patient ID {patient_id}, but 'patient_info.csv' does not exist.")
    return redirect(url_for('home'))

@app.route('/select/<int:patient_id>')
def select(patient_id):
    # If the patient ID is different from the one in the session, clear the session data
    if 'patient_id' in session and session['patient_id'] != patient_id:
        session.clear()

    # Store the selected patient ID in the session
    session['patient_id'] = patient_id  # Store as integer

    # Initialize session variables if they don't exist
    session.setdefault('questionnaire_data', {})
    session.setdefault('completed_questionnaires', [])

    # Check if a reset has just been done
    if session.get('is_reset', False):
        # After reset, ensure 'completed_questionnaires' is empty
        session['questionnaire_data'] = {}
        session['completed_questionnaires'] = []
        app.logger.debug(f"Session has been reset for patient ID: {patient_id}")
        # Do not pop 'is_reset'; let 'questionnaire_route' handle it
    elif not session['questionnaire_data'] and os.path.exists('patient_info.csv'):
        # Only load from CSV if not just reset and session data is empty
        patient_data = pd.read_csv('patient_info.csv', dtype=str)  # Read all as strings
        # Find the patient data
        patient_row = patient_data.loc[patient_data['id'] == str(patient_id)]
        if not patient_row.empty:
            # Get questionnaire answers
            patient_answers = patient_row.to_dict(orient='records')[0]

            # Extract questionnaire data from patient_answers
            questionnaire_data = session.get('questionnaire_data', {})
            completed_questionnaires = session.get('completed_questionnaires', [])
            for q_name in questionnaires.keys():
                # Standardize questionnaire name
                q_name_lower = q_name.lower()
                # Get question IDs for this questionnaire
                question_ids = [
                    question['id']
                    for section in questionnaires[q_name_lower]['sections']
                    for question in section['questions']
                    if 'id' in question
                ]
                # Collect answers for these question IDs
                answers = {}
                for qid in question_ids:
                    value = patient_answers.get(qid, '')
                    if pd.isna(value):
                        value = ''
                    value = str(value).strip()
                    if value.lower() != 'nan' and value != '':
                        answers[qid] = value
                    else:
                        answers[qid] = ''
                # Only mark as complete if at least one answer is provided
                if any(answers.values()):
                    questionnaire_data[q_name_lower] = answers
                    # Mark questionnaire as completed if not already marked
                    if q_name_lower not in completed_questionnaires:
                        completed_questionnaires.append(q_name_lower)
            # Update session with questionnaire data and completed questionnaires
            session['questionnaire_data'] = questionnaire_data
            session['completed_questionnaires'] = completed_questionnaires
            app.logger.debug(f"Loaded questionnaire data for patient ID {patient_id}: {questionnaire_data}")
        else:
            app.logger.warning(f"No patient found with ID {patient_id} in 'patient_info.csv'.")

    # Get the list of completed questionnaires from the session
    completed_questionnaires = session.get('completed_questionnaires', [])
    app.logger.debug(f"Patient ID: {patient_id}")
    app.logger.debug(f"Completed Questionnaires: {completed_questionnaires}")
    app.logger.debug(f"Questionnaire Data in Session: {session.get('questionnaire_data', {})}")

    # Pass 'questionnaire_data' to the template
    return render_template(
        'select.html',
        questionnaires=questionnaires,
        completed_questionnaires=completed_questionnaires,
        questionnaire_data=session.get('questionnaire_data', {})
    )

@app.route('/questionnaire/<questionnaire_name>', methods=['GET', 'POST'])
@nocache
def questionnaire_route(questionnaire_name):
    questionnaire_name = questionnaire_name.lower()
    if questionnaire_name not in [q.lower() for q in questionnaires.keys()]:
        app.logger.error(f"Questionnaire '{questionnaire_name}' not found.")
        return "Questionnaire not found.", 404

    # Find the actual questionnaire name (case-insensitive match)
    actual_questionnaire_name = next((q for q in questionnaires.keys() if q.lower() == questionnaire_name), None)
    if not actual_questionnaire_name:
        app.logger.error(f"Questionnaire '{questionnaire_name}' not found after case-insensitive match.")
        return "Questionnaire not found.", 404
    questionnaire_data = questionnaires[actual_questionnaire_name]

    if request.method == 'POST':
        form_data = request.form.to_dict()
        # Store form data in session, overwriting previous answers for this questionnaire
        questionnaire_session_data = session.get('questionnaire_data', {})
        questionnaire_session_data[questionnaire_name] = form_data
        session['questionnaire_data'] = questionnaire_session_data  # Reassign to session

        # Update completed questionnaires
        completed_questionnaires = session.get('completed_questionnaires', [])
        if questionnaire_name not in completed_questionnaires:
            completed_questionnaires.append(questionnaire_name)
            session['completed_questionnaires'] = completed_questionnaires  # Reassign to session

        app.logger.debug(f"Submitted data for '{questionnaire_name}': {form_data}")
        app.logger.debug(f"Session Data after submission: {session}")

        return redirect(url_for('select', patient_id=session['patient_id']))

    # GET request: Load previous answers
    previous_answers = session.get('questionnaire_data', {}).get(questionnaire_name, {})
    if not previous_answers:
        if session.get('is_reset', False):
            # If a reset was done, do not load answers from CSV
            previous_answers = {}
            session.pop('is_reset')  # Remove the reset flag
            app.logger.debug(f"Reset done. Not loading previous answers for '{questionnaire_name}'.")
        elif os.path.exists('patient_info.csv'):
            patient_id = session.get('patient_id')
            if patient_id is not None:
                patient_data = pd.read_csv('patient_info.csv', dtype=str)  # Read all as strings
                patient_row = patient_data.loc[patient_data['id'] == str(patient_id)]
                if not patient_row.empty:
                    patient_answers = patient_row.to_dict(orient='records')[0]
                    # Extract answers for this questionnaire
                    question_ids = []
                    for section in questionnaire_data['sections']:
                        for question in section['questions']:
                            if 'id' in question:
                                question_ids.append(question['id'])
                    # Build previous_answers, handling 'nan' and empty strings
                    previous_answers = {}
                    for qid in question_ids:
                        value = patient_answers.get(qid, '')
                        if pd.isna(value):
                            value = ''
                        value = str(value).strip()
                        if value.lower() != 'nan' and value != '':
                            previous_answers[qid] = value
                        else:
                            previous_answers[qid] = ''
                    # Update session with these answers
                    questionnaire_session_data = session.get('questionnaire_data', {})
                    questionnaire_session_data[questionnaire_name] = previous_answers
                    session['questionnaire_data'] = questionnaire_session_data
                    app.logger.debug(f"Loaded previous answers for '{questionnaire_name}': {previous_answers}")

    app.logger.debug(f"Rendering questionnaire '{questionnaire_name}' with previous answers: {previous_answers}")
    return render_template('questionnaire.html', questionnaire=questionnaire_data, previous_answers=previous_answers, questionnaire_name=questionnaire_name)

@app.route('/predict')
def predict():
    # Get the list of completed questionnaires
    completed_questionnaires = session.get('completed_questionnaires', [])
    if not completed_questionnaires:
        error = 'Please complete at least one questionnaire before predicting.'
        app.logger.warning("Prediction attempted without completed questionnaires.")
        return render_template('select.html', questionnaires=questionnaires, completed_questionnaires=[], error=error)

    # Load the appropriate model
    model, feature_names = load_model(completed_questionnaires)
    if model is None:
        error = 'Model not found for the selected questionnaires.'
        app.logger.error("Prediction attempted without a valid model.")
        return render_template('select.html', questionnaires=questionnaires, completed_questionnaires=completed_questionnaires, error=error)

    # Log feature names
    app.logger.debug(f"Feature Names from Model: {feature_names}")

    # Collect the form data
    questionnaire_data = session.get('questionnaire_data', {})
    app.logger.debug(f"Questionnaire Data from Session: {questionnaire_data}")

    features = []
    for feature in feature_names:
        value = '0'  # Default value
        for q in completed_questionnaires:
            q_lower = q.lower()
            # Get question IDs for the questionnaire
            criteria = [question['id'] for section in questionnaires[q_lower]['sections'] for question in section['questions'] if 'id' in question]
            if feature in criteria:
                value = questionnaire_data.get(q_lower, {}).get(feature, '0').strip()
                break
        if value.lower() == 'nan' or value == '':
            value = '0'
        try:
            float_val = float(value)
            features.append(float_val)
            app.logger.debug(f"Feature: {feature}, Value: {float_val}")
        except ValueError:
            error = f"Invalid input for {feature}. Please enter a valid number."
            app.logger.error(f"ValueError for feature '{feature}': {value}")
            return render_template('select.html', questionnaires=questionnaires, completed_questionnaires=completed_questionnaires, error=error)

    # Create a DataFrame with the features and feature names
    features_df = pd.DataFrame([features], columns=feature_names)
    app.logger.debug(f"Features DataFrame for Prediction:\n{features_df}")

    # Make the prediction
    prediction = model.predict(features_df)
    predicted_diagnosis = diagnosis_map.get(prediction[0], 'Unknown')

    # Get prediction probabilities
    probabilities = model.predict_proba(features_df)[0]
    # Map probabilities to diagnosis labels
    prob_dict = {}
    for idx, prob in enumerate(probabilities):
        diagnosis_label = diagnosis_map.get(idx, 'Unknown')
        prob_dict[diagnosis_label] = round(prob * 100, 2)  # Convert to percentage and round off

    app.logger.debug(f"Prediction: {predicted_diagnosis}")
    app.logger.debug(f"Probabilities: {prob_dict}")

    # Save the prediction and questionnaire answers to patient_info.csv
    patient_id = session.get('patient_id')  # Ensure patient_id is retrieved from the session
    if patient_id is not None:
        # Collect all questionnaire answers
        all_answers = {}
        for q_name, answers in questionnaire_data.items():
            # Clean up answers to remove empty strings and 'nan'
            clean_answers = {}
            for key, value in answers.items():
                if isinstance(value, str) and value.lower() != 'nan' and value.strip() != '':
                    clean_answers[key] = value.strip()
                else:
                    clean_answers[key] = ''
            all_answers.update(clean_answers)

        # Prepare data to save
        data_to_save = {
            'id': str(patient_id),  # Ensure patient ID is a string
            'diagnosis': predicted_diagnosis,
            'prob_Normal': str(prob_dict.get('Normal', 0)),
            'prob_MCI': str(prob_dict.get('Mild Cognitive Impairment', 0)),
            'prob_AD': str(prob_dict.get("Alzheimer's Disease", 0)),
        }
        data_to_save.update(all_answers)

        # Load existing patient data
        if os.path.exists('patient_info.csv'):
            patient_data = pd.read_csv('patient_info.csv', dtype=str)  # Read all as strings
        else:
            # Initialize with only the necessary columns if CSV doesn't exist
            patient_data = pd.DataFrame(columns=['id', 'diagnosis', 'prob_Normal', 'prob_MCI', 'prob_AD'])

        # Check if patient exists
        if str(patient_id) in patient_data['id'].values:
            # Locate the patient row
            patient_index = patient_data.index[patient_data['id'] == str(patient_id)].tolist()[0]
            # Update existing fields
            for key, value in data_to_save.items():
                if key in patient_data.columns:
                    patient_data.at[patient_index, key] = value
                else:
                    # Add new column if it doesn't exist
                    patient_data[key] = ''
                    patient_data.at[patient_index, key] = value
        else:
            # Append new patient data
            patient_data = patient_data.append(data_to_save, ignore_index=True)

        # Save back to CSV
        patient_data.to_csv('patient_info.csv', index=False)

        app.logger.debug("Data saved to patient_info.csv:")
        app.logger.debug(patient_data.loc[patient_data['id'] == str(patient_id)].to_dict(orient='records'))

    # Clear the session data if you want to reset after prediction
    # session.clear()

    return render_template('result.html', prediction=predicted_diagnosis, probabilities=prob_dict)

@app.route('/reset')
def reset():
    patient_id = session.get('patient_id')
    app.logger.debug(f"Resetting session for patient ID: {patient_id}")
    session.clear()
    if patient_id is not None:
        # Reinitialize session variables for the patient
        session['patient_id'] = patient_id
        session['questionnaire_data'] = {}
        session['completed_questionnaires'] = []
        session['is_reset'] = True  # Add flag to indicate a reset has occurred
        app.logger.debug(f"Session reset for patient ID: {patient_id}")
        return redirect(url_for('select', patient_id=patient_id))
    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
