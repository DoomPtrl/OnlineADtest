from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

# Diagnosis mapping (replace with your actual diagnosis labels)
diagnosis_map = {
    0: 'Normal',
    1: 'Mild Cognitive Impairment',
    2: 'Alzheimer\'s Disease'
}

# Define the questionnaires and their criteria
questionnaires = {
    'mmse': {
        'name': 'MMSE',
        'criteria': ['MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY',
                     'MMSEASON', 'MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE',
                     'WORD1', 'WORD2', 'WORD3', 'MMD', 'MML', 'MMR', 'MMO', 'MMW', 'WORD1DL', 'WORD2DL', 'WORD3DL',
                     'MMWATCH', 'MMPENCIL', 'MMREPEAT', 'MMHAND', 'MMFOLD', 'MMONFLR',
                     'MMREAD', 'MMWRITE', 'MMDRAW'],
        'criteria_range': [[0,1] for _ in range(30)],
        'questions': ['What is the date today?', 'What year is it?', 'What month is it?', 'What day of the week is it?',
                      'What season is it?', 'What hospital are we in?', 'What floor are we on?', 'What city are we in?','What area are we in?', 'What state are we in?',
                      'Word 1 recall', 'Word 2 recall', 'Word 3 recall', 'D', 'L', 'R', 'O', 'W', 'Word 1 delayed recall', 'Word 2 delayed recall', 'Word 3 delayed recall',
                      'Identify a watch', 'Identify a pencil', 'Repeat "No ifs, ands, or buts"', 'Follow a three-step command (hand)', 'Follow a three-step command (fold)', 'Follow a three-step command (on floor)',
                      'Read and obey "Close your eyes"', 'Write a sentence', 'Copy a design']
    },
    'moca': {
        'name': 'MOCA',
        'criteria': ["TRAILS", "CUBE", "CLOCKCON", "CLOCKNO", "CLOCKHAN",
                     "LION", "RHINO", "CAMEL",
                     "IMMT1W1", "IMMT1W2", "IMMT1W3", "IMMT1W4", "IMMT1W5",
                     "IMMT2W1", "IMMT2W2", "IMMT2W3", "IMMT2W4", "IMMT2W5",
                     "DIGFOR", "DIGBACK",
                     "LETTERS",
                     "SERIAL1", "SERIAL2", "SERIAL3", "SERIAL4", "SERIAL5",
                     "REPEAT1", "REPEAT2",
                     "FFLUENCY",
                     "ABSTRAN", "ABSMEAS",
                     "DELW1", "DELW2", "DELW3", "DELW4", "DELW5",
                     "DATE", "MONTH", "YEAR", "DAY", "PLACE", "CITY"],
        'criteria_range': [[0,1] for _ in range(42)],
        'questions': ['Draw a line between the numbers and letters in order', 'Copy the cube', 'Draw a clock contour', 'Place the numbers on the clock', 'Set the hands on the clock',
                      'Name the animal: lion', 'Name the animal: rhinoceros', 'Name the animal: camel',
                      'Immediate recall word 1', 'Immediate recall word 2', 'Immediate recall word 3', 'Immediate recall word 4', 'Immediate recall word 5',
                      'Delayed recall cue 1', 'Delayed recall cue 2', 'Delayed recall cue 3', 'Delayed recall cue 4', 'Delayed recall cue 5',
                      'Digit span forward', 'Digit span backward',
                      'Tap when you hear the letter A', 'Serial subtraction 1', 'Serial subtraction 2', 'Serial subtraction 3', 'Serial subtraction 4', 'Serial subtraction 5',
                      'Repeat sentence 1', 'Repeat sentence 2',
                      'Phonemic fluency (words starting with F)',
                      'Abstraction (train-bicycle)', 'Abstraction (watch-ruler)',
                      'Delayed recall word 1', 'Delayed recall word 2', 'Delayed recall word 3', 'Delayed recall word 4', 'Delayed recall word 5',
                      'What is the date today?', 'What is the month?', 'What is the year?', 'What day of the week is it?', 'What place are we in?', 'What city are we in?']
    },
    'npiq': {
        'name': 'NPIQ',
        'criteria': ['NPI' + chr(i) + 'SEV' for i in range(65, 65 + 12)],
        'criteria_range': [[0, 1, 2, 3] for _ in range(12)],
        'questions': ['Delusions', 'Hallucinations', 'Agitation/Aggression', 'Depression/Dysphoria', 'Anxiety', 'Elation/Euphoria', 'Apathy/Indifference', 'Disinhibition', 'Irritability/Lability', 'Motor Disturbances', 'Night-time Behaviors', 'Appetite/Eating']
    }
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
            # Redirect to input form with selected questionnaires
            return redirect(url_for('input_data', questionnaires=','.join(selected_questionnaires)))
    else:
        return render_template('index.html', questionnaires=questionnaires)

@app.route('/input', methods=['GET', 'POST'])
def input_data():
    selected_questionnaires = request.args.get('questionnaires').split(',')
    # Load the model and feature names
    model, feature_names = load_model(selected_questionnaires)
    if model is None:
        error = 'Model not found for the selected questionnaires.'
        return render_template('input.html', error=error)

    if request.method == 'POST':
        # Collect feature values
        features = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value == '' or value is None:
                value = 0  # Or handle as appropriate
            features.append(float(value))

        # Ensure the feature array is in the correct order
        features = np.array(features).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(features)
        predicted_diagnosis = diagnosis_map.get(prediction[0], 'Unknown')

        return render_template('result.html', prediction=predicted_diagnosis)
    else:
        # Prepare features_info with feature names, questions, and their ranges
        features_info = []
        for feature in feature_names:
            if feature in feature_info_map:
                features_info.append({
                    'feature': feature,  # Used as the name attribute in form inputs
                    'question': feature_info_map[feature]['question'],
                    'range': feature_info_map[feature]['range']
                })
            else:
                # Handle cases where the feature is not found
                pass
        return render_template('input.html', features_info=features_info, questionnaires=selected_questionnaires)

if __name__ == '__main__':
    app.run(debug=True)
