from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import os
from questionnaire_data import questionnaires
import pandas as pd
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'  # Replace with a secure random key

# Diagnosis mapping
diagnosis_map = {
    0: 'Normal',
    1: 'Mild Cognitive Impairment',
    2: "Alzheimer's Disease"
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    session.clear()  # Clear the session data when the user goes back to the home page
    # Assuming client data is stored in 'client_data.csv'
    client_data = pd.read_csv('client_data.csv')
    # Convert DataFrame to a list of dictionaries for easier template rendering
    client_list = client_data.to_dict(orient='records')
    return render_template('client_info.html', clients=client_list)

@app.route('/add_client', methods=['POST'])
def add_client():
    # Get form data
    new_client_data = request.form.to_dict()
    client_data = pd.read_csv('client_data.csv')
    new_client_data['id'] = int(client_data['id'].max() + 1) if not client_data.empty else 1
    new_client_data['age'] = int(new_client_data['age'])
    new_client_data['diagnosis'] = 'Unknown'

    # Handle file upload
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo and allowed_file(photo.filename):
            filename = secure_filename(photo.filename)
            # Rename the file to include client ID
            filename = f"client_{new_client_data['id']}_{filename}"
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_client_data['photo'] = filename
        else:
            new_client_data['photo'] = 'default.png'  # Use a default image
    else:
        new_client_data['photo'] = 'default.png'

    # Save the new client data
    new_client_df = pd.DataFrame([new_client_data])
    client_data = pd.concat([client_data, new_client_df], ignore_index=True)
    client_data.to_csv('client_data.csv', index=False)

    return redirect(url_for('home'))


@app.route('/edit_client/<int:client_id>', methods=['GET', 'POST'])
def edit_client(client_id):
    client_data = pd.read_csv('client_data.csv')
    if request.method == 'POST':
        # Update client information
        updated_client_data = request.form.to_dict()
        updated_client_data['id'] = client_id
        updated_client_data['age'] = int(updated_client_data['age'])

        # Handle file upload
        if 'photo' in request.files and request.files['photo'].filename != '':
            photo = request.files['photo']
            if photo and allowed_file(photo.filename):
                filename = secure_filename(photo.filename)
                filename = f"client_{client_id}_{filename}"
                photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                updated_client_data['photo'] = filename
            else:
                # Keep the existing photo if upload fails
                existing_photo = client_data.loc[client_data['id'] == client_id, 'photo'].values[0]
                updated_client_data['photo'] = existing_photo
        else:
            # Keep the existing photo if no new photo is uploaded
            existing_photo = client_data.loc[client_data['id'] == client_id, 'photo'].values[0]
            updated_client_data['photo'] = existing_photo

        # Update the client data
        client_data.loc[client_data['id'] == client_id, ['name', 'sex', 'age', 'address', 'photo']] = [
            updated_client_data['name'],
            updated_client_data['sex'],
            updated_client_data['age'],
            updated_client_data['address'],
            updated_client_data['photo']
        ]
        client_data.to_csv('client_data.csv', index=False)
        return redirect(url_for('home'))
    else:
        # Render the edit form with current client data
        client = client_data.loc[client_data['id'] == client_id].to_dict(orient='records')[0]
        return render_template('edit_client.html', client=client)


@app.route('/delete_client/<int:client_id>', methods=['POST'])
def delete_client(client_id):
    client_data = pd.read_csv('client_data.csv')
    # Remove the client with the given ID
    client_data = client_data[client_data['id'] != client_id]
    # Save the updated data
    client_data.to_csv('client_data.csv', index=False)
    return redirect(url_for('home'))

@app.route('/select/<int:client_id>')
def select(client_id):
    # Store the selected client ID in the session
    session['client_id'] = client_id

    # Get the list of completed questionnaires from the session
    completed_questionnaires = session.get('completed_questionnaires', [])
    return render_template('select.html', questionnaires=questionnaires, completed_questionnaires=completed_questionnaires, back_to_client_info=True)

@app.route('/<questionnaire>', methods=['GET', 'POST'])
def questionnaire_route(questionnaire):
    if questionnaire not in questionnaires:
        return "Questionnaire not found.", 404

    template_name = f"{questionnaire}_input.html"

    if request.method == 'POST':
        form_data = {key: value for key, value in request.form.to_dict().items()}
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

        # Redirect back to the select page
        return redirect(url_for('select', client_id=session['client_id']))

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
        return render_template('select.html', questionnaires=questionnaires, completed_questionnaires=[], error=error)

    # Load the appropriate model
    model, feature_names = load_model(completed_questionnaires)
    if model is None:
        error = 'Model not found for the selected questionnaires.'
        return render_template('select.html', questionnaires=questionnaires, completed_questionnaires=completed_questionnaires, error=error)

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
            return render_template('select.html', error=error)

    # Create a DataFrame with the features and feature names
    features_df = pd.DataFrame([features], columns=feature_names)

    # Make the prediction
    print(f"Features for prediction:\n{features_df}")  # Debugging print statement
    prediction = model.predict(features_df)
    predicted_diagnosis = diagnosis_map.get(prediction[0], 'Unknown')

    # Save the prediction to client_data.csv
    client_id = session.get('client_id')  # Ensure client_id is retrieved from the session
    if client_id is not None:
        client_data = pd.read_csv('client_data.csv')
        client_data.loc[client_data['id'] == int(client_id), 'diagnosis'] = predicted_diagnosis
        client_data.to_csv('client_data.csv', index=False)

    # Clear the session data if you want to reset after prediction
    # session.clear()

    return render_template('result.html', prediction=predicted_diagnosis)

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('select'))

if __name__ == '__main__':
    app.run(debug=True)
