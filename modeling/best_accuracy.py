import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelBinarizer

# Function for loading and preprocessing data
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
        diagnosis = pd.merge_asof(
            diagnosis.sort_values(['VISDATE', 'PTID']),
            data.sort_values(['VISDATE', 'PTID']),
            on='VISDATE',
            by='PTID',
            direction='nearest',
            tolerance=pd.Timedelta('450D')
        ).dropna()
    
    # Prepare features and labels
    X = diagnosis[[item for sublist in criterias_combo for item in sublist]]
    y = diagnosis['DIAGNOSIS']
    return X, y

# Define the models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42, algorithm='SAMME'),
    'ExtraTrees': ExtraTreesClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)  # Added Logistic Regression
}

# Define hyperparameters for GridSearch
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    'ExtraTrees': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    'LogisticRegression': {  # Hyperparameters for Logistic Regression
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']
    }
}

# Iterate over each combination of data
combinations = ['mmse', 'moca', 'npiq', 'mmse_moca', 'moca_npiq', 'mmse_npiq', 'all']

all_model_results = []

for combo in combinations:
    print(f"\nEvaluating combination: {combo}")
    
    # Load data
    X, y = load_combination_data(combo)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    for model_name, model in models.items():
        print(f"  Evaluating model: {model_name}")
        
        # Define GridSearchCV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=5,
            n_jobs=-1,
            scoring='accuracy',
            verbose=1
        )
        grid.fit(X_train, y_train)
        
        # Cross-validation score
        cv_score = grid.best_score_
        print(f"    Best CV Score: {cv_score:.4f}")
        
        # Evaluate on test set
        y_pred = grid.best_estimator_.predict(X_test)
        
        # Determine if it's binary or multi-class classification
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        
        if num_classes == 2:
            # Binary classification
            y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob)
        else:
            # Multi-class classification
            y_prob = grid.best_estimator_.predict_proba(X_test)
            # Binarize the output for multi-class ROC AUC
            lb = LabelBinarizer()
            lb.fit(y_test)
            y_test_binarized = lb.transform(y_test)
            if y_test_binarized.shape[1] == 1:
                y_test_binarized = np.hstack([1 - y_test_binarized, y_test_binarized])
            test_auc = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr')
        
        # Compute test accuracy
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Compute classification report
        class_report = classification_report(y_test, y_pred, zero_division=0)
        
        print(f"    Test Accuracy: {test_accuracy:.4f}")
        print(f"    Test ROC-AUC: {test_auc:.4f}")
        print(f"    Best Parameters: {grid.best_params_}")
        print(f"    Classification Report:\n{class_report}")
        
        # Store all model results
        all_model_results.append({
            'Combination': combo,
            'Model': model_name,
            'CV_Score': cv_score,
            'Test_Accuracy': test_accuracy,
            'Test_ROC_AUC': test_auc,
            'Best_Params': grid.best_params_
        })

# Print summary of all models for each combination in a table
all_results_df = pd.DataFrame(all_model_results)
print("\nSummary of All Model Results:")
print(all_results_df)

# Optionally save the results to a CSV file
all_results_df.to_csv('all_model_results.csv', index=False)
