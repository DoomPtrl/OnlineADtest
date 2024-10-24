import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Iterate over each combination of data
combinations = ['mmse', 'moca', 'npiq', 'mmse_moca', 'moca_npiq', 'mmse_npiq', 'all']

best_model_results = {}

for combo in combinations:
    print(f"Evaluating combination: {combo}")
    
    # Load data
    X, y = load_combination_data(combo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"    Test Accuracy: {test_accuracy:.4f}")
    
    # Store the best model
    best_model_results[combo] = {'model': model, 'accuracy': test_accuracy}
    print(f"Best model for combination {combo}: Neural Network with accuracy {test_accuracy:.4f}\n")

# Print summary of best models for each combination
for combo, result in best_model_results.items():
    print(f"Combination: {combo}, Best Model: Neural Network, Accuracy: {result['accuracy']:.4f}")
