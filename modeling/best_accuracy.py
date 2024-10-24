import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Function for loading and preprocessing data (using the load_and_preprocess function from pipeline.py)
def load_combination_data(combo_name):
    diagnosis_path = 'modeling/data/diagnosis.csv'  # Replace with the actual path to the diagnosis data
    
    data_paths = {
        'mmse': 'modeling/cleaned_data/mmse_clean.csv',  # Replace with actual paths to MMSE, MOCA, and NPI-Q data
        'moca': 'modeling/cleaned_data/moca_clean.csv',
        'npiq': 'modeling/cleaned_data/npiq_clean.csv'

    }
    
    criterias = {
        'mmse': ['MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY', 'MMSEASON', 'MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE',
                 'WORD1', 'WORD2', 'WORD3', 'MMD', 'MML', 'MMR', 'MMO', 'MMW', 'WORD1DL', 'WORD2DL', 'WORD3DL',
                 'MMWATCH', 'MMPENCIL', 'MMREPEAT', 'MMHAND', 'MMFOLD', 'MMONFLR', 'MMREAD', 'MMWRITE', 'MMDRAW'],
        'moca': ['TRAILS', 'CUBE', 'CLOCKCON', 'CLOCKNO', 'CLOCKHAN', 'LION', 'RHINO', 'CAMEL',
                 'IMMT1W1', 'IMMT1W2', 'IMMT1W3', 'IMMT1W4', 'IMMT1W5', 'IMMT2W1', 'IMMT2W2', 'IMMT2W3', 'IMMT2W4', 'IMMT2W5',
                 'DIGFOR', 'DIGBACK', 'LETTERS', 'SERIAL1', 'SERIAL2', 'SERIAL3', 'SERIAL4', 'SERIAL5',
                 'REPEAT1', 'REPEAT2', 'FFLUENCY', 'ABSTRAN', 'ABSMEAS', 'DELW1', 'DELW2', 'DELW3', 'DELW4', 'DELW5',
                 'DATE', 'MONTH', 'YEAR', 'DAY', 'PLACE', 'CITY'],
        'npiq': ['NPIASEV', 'NPIBSEV', 'NPICSEV', 'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 'NPIHSEV', 'NPIISEV', 'NPIJSEV', 'NPIKSEV', 'NPILSEV']
    }
    
    if combo_name in ['mmse', 'moca', 'npiq']:
        data_paths_combo = [data_paths[combo_name]]
        criterias_combo = [criterias[combo_name]]
    elif combo_name == 'mmse_moca':
        data_paths_combo = [data_paths['mmse'], data_paths['moca']]
        criterias_combo = [criterias['mmse'], criterias['moca']]
    elif combo_name == 'moca_npiq':
        data_paths_combo = [data_paths['moca'], data_paths['npiq']]
        criterias_combo = [criterias['moca'], criterias['npiq']]
    elif combo_name == 'mmse_npiq':
        data_paths_combo = [data_paths['mmse'], data_paths['npiq']]
        criterias_combo = [criterias['mmse'], criterias['npiq']]
    elif combo_name == 'all':
        data_paths_combo = [data_paths['mmse'], data_paths['moca'], data_paths['npiq']]
        criterias_combo = [criterias['mmse'], criterias['moca'], criterias['npiq']]
    else:
        raise ValueError(f"Unknown combination name: {combo_name}")
    
    # Load and preprocess the data using the criteria and merge it with the diagnosis data
    datas = [pd.read_csv(path)[criteria + ['VISDATE', 'PTID']].dropna() for path, criteria in zip(data_paths_combo, criterias_combo)]
    diagnosis = pd.read_csv(diagnosis_path)[['EXAMDATE', 'PTID', 'DIAGNOSIS']].rename(columns={'EXAMDATE': 'VISDATE'}).dropna()
    diagnosis['DIAGNOSIS'] = diagnosis['DIAGNOSIS'] - 1  # Adjusting diagnosis labels if needed
    diagnosis['VISDATE'] = pd.to_datetime(diagnosis['VISDATE'])
    
    # Merging diagnosis data with each of the questionnaire data based on the nearest visit date within a 1-year tolerance
    for data in datas:
        data['VISDATE'] = pd.to_datetime(data['VISDATE'])
        diagnosis = pd.merge_asof(diagnosis.sort_values(['VISDATE', 'PTID']), data.sort_values(['VISDATE', 'PTID']), on='VISDATE', by='PTID', direction='nearest', tolerance=pd.Timedelta('365D')).dropna()
    
    # Prepare features and labels
    X = diagnosis[[item for sublist in criterias_combo for item in sublist]]
    y = diagnosis['DIAGNOSIS']
    return X, y

# Define the model pipelines
models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'SVM': SVC(),
}

# Define hyperparameters for GridSearch
param_grids = {
    'RandomForest': {'classifier__n_estimators': [50, 100, 150], 'classifier__max_depth': [None, 10, 20]},
    'GradientBoosting': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1, 0.2]},
    'AdaBoost': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1, 1]},
    'ExtraTrees': {'classifier__n_estimators': [50, 100, 150], 'classifier__max_depth': [None, 10, 20]},
    'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['rbf', 'linear']},
}

# Iterate over each combination of data
combinations = ['mmse', 'moca', 'npiq', 'mmse_moca', 'moca_npiq', 'mmse_npiq', 'all']

all_model_results = []

for combo in combinations:
    print(f"Evaluating combination: {combo}")
    
    # Load data
    X, y = load_combination_data(combo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for model_name, model in models.items():
        print(f"  Evaluating model: {model_name}")
        
        # Define pipeline (scaling only needed for SVM)
        if model_name == 'SVM':
            pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
        else:
            pipeline = Pipeline([('classifier', model)])
        
        # GridSearch for best hyperparameters
        grid = GridSearchCV(pipeline, param_grids[model_name], cv=5, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)
        
        # Cross-validation score
        cv_score = grid.best_score_
        print(f"    CV Score: {cv_score:.4f}")
        
        # Evaluate on test set
        y_pred = grid.best_estimator_.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"    Test Accuracy: {test_accuracy:.4f}")
        
        # Store all model results
        all_model_results.append({'Combination': combo, 'Model': model_name, 'CV_Score': cv_score, 'Test_Accuracy': test_accuracy})

# Print summary of all models for each combination in a table
all_results_df = pd.DataFrame(all_model_results)
print(all_results_df)

# Optionally save the results to a CSV file
all_results_df.to_csv('all_model_results.csv', index=False)
