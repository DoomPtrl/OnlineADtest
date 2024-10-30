# data_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Function for loading and preprocessing data
def load_combination_data(combo_name, tolerance_days=450):
    diagnosis_path = 'modeling/data/diagnosis.csv'  # Update path as necessary
    
    data_paths = {
        'mmse': 'modeling/cleaned_data/mmse_clean.csv',
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
    
    # Merge diagnosis data with each of the questionnaire data based on the nearest visit date within a tolerance
    for data in datas:
        data['VISDATE'] = pd.to_datetime(data['VISDATE'])
        diagnosis = pd.merge_asof(
            diagnosis.sort_values(['VISDATE', 'PTID']),
            data.sort_values(['VISDATE', 'PTID']),
            on='VISDATE',
            by='PTID',
            direction='nearest',
            tolerance=pd.Timedelta(f'{tolerance_days}D')
        ).dropna()
    
    # Prepare features and labels
    X = diagnosis[[item for sublist in criterias_combo for item in sublist]]
    y = diagnosis['DIAGNOSIS']
    
    # Optional: Encode categorical variables if any (assuming all are numerical or binary)
    # If there are categorical variables, apply encoding here.
    # Example:
    # X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def save_preprocessed_data(combinations, output_dir='modeling/combination_data'):
    """
    Preprocess and save data for each combination.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for combo in combinations:
        print(f"Processing combination: {combo}")
        X, y = load_combination_data(combo)
        
        # Combine features and target for saving
        data_to_save = pd.concat([pd.DataFrame(X), pd.Series(y, name='DIAGNOSIS')], axis=1)
        
        # Define the output file path
        output_file = os.path.join(output_dir, f"{combo}_preprocessed.csv")
        
        # Save to CSV
        data_to_save.to_csv(output_file, index=False)
        print(f"Saved preprocessed data to {output_file}\n")

if __name__ == "__main__":
    # Define all combinations
    combinations = ['mmse', 'moca', 'npiq', 'mmse_moca', 'moca_npiq', 'mmse_npiq', 'all']
    
    # Save preprocessed data
    save_preprocessed_data(combinations)
