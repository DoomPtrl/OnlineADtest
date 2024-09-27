import pandas as pd
from sklearn.model_selection import train_test_split
from bayesian import *
from sklearn.metrics import accuracy_score

CRITERIA_MMSE = ['MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY',
       'MMSEASON', 'MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE',
        'WORD1', 'WORD2', 'WORD3', 'MMD', 'MML', 'MMR',
       'MMO', 'MMW', 'WORD1DL', 'WORD2DL', 'WORD3DL',
       'MMWATCH', 'MMPENCIL', 'MMREPEAT', 'MMHAND', 'MMFOLD', 'MMONFLR',
       'MMREAD', 'MMWRITE', 'MMDRAW']
CRITERIA_NPIQ = ['NPI'+chr(i)+'SEV' for i in range(65, 65+12)]
CRITERIA_MOCA = ["TRAILS","CUBE",
            "CLOCKCON","CLOCKNO","CLOCKHAN",
            "LION","RHINO","CAMEL",
            "IMMT1W1","IMMT1W2","IMMT1W3","IMMT1W4","IMMT1W5","IMMT2W1","IMMT2W2","IMMT2W3","IMMT2W4","IMMT2W5",
            "DIGFOR","DIGBACK",
            "LETTERS",
            "SERIAL1","SERIAL2","SERIAL3","SERIAL4","SERIAL5",
            "REPEAT1","REPEAT2",
            "FFLUENCY",
            "ABSTRAN","ABSMEAS",
            "DELW1","DELW2","DELW3","DELW4","DELW5",
            "DATE","MONTH","YEAR","DAY","PLACE","CITY"]

# Function to load and preprocess data
def load_and_preprocess(data_paths, criterias, diagnosis_path, balance=False):
    if type(data_paths) is not list:
        data_paths = [data_paths]
    if type(criterias) is not list:
        criterias = [criterias]
    datas=[pd.read_csv(path)[criteria+['VISDATE','PTID']].dropna() for path, criteria in zip(data_paths, criterias)]
    diagnosis = pd.read_csv(diagnosis_path)[['EXAMDATE', 'PTID', 'DIAGNOSIS']].rename(columns={'EXAMDATE': 'VISDATE'}).dropna()
    diagnosis['DIAGNOSIS'] = diagnosis['DIAGNOSIS'] - 1  # Adjusting diagnosis labels if needed
    diagnosis['VISDATE'] = pd.to_datetime(diagnosis['VISDATE'])
    for data in datas:
        data['VISDATE'] = pd.to_datetime(data['VISDATE'])
        diagnosis = pd.merge_asof(diagnosis.sort_values(['VISDATE', 'PTID']), data.sort_values(['VISDATE', 'PTID']), on='VISDATE', by='PTID', direction='nearest', tolerance=pd.Timedelta('365D')).dropna()
    if balance:
        diagnosis = diagnosis.groupby('DIAGNOSIS').apply(lambda x: x.sample(diagnosis['DIAGNOSIS'].value_counts().min())).reset_index(drop=True)
    X=diagnosis[[item for sublist in criterias for item in sublist]]
    y=diagnosis['DIAGNOSIS']
    return X,y

# Function to train and evaluate a classifier
# def train_and_evaluate(datas, criterias, max_scores, class_priors=None):
#     if type(datas) is not list:
#         datas = [datas]
#     if type(criterias) is not list:
#         criterias = [criterias]
#     if type(max_scores) is not list:
#         max_scores = [max_scores]
#     X_train, X_test, y_train, y_test = train_test_split(data[criterias], data['DIAGNOSIS'], test_size=0.2, random_state=42)
#     params
#     y_pred = bayesian_classifier(X_test, weights, criterias, max_scores, class_priors)
#     return accuracy_score(y_test, y_pred)

if __name__=='__main__':
# Define the questionnaires
    import pandas as pd
    import itertools
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    questionnaires = [
        ('mmse', 'data/mmse.csv', CRITERIA_MMSE),
        ('moca', 'data/moca.csv', CRITERIA_MOCA),
        ('npiq', 'data/npiq.csv', CRITERIA_NPIQ),
    ]

    diagnosis_path = 'data/diagnosis.csv'

    # Iterate over all combinations
    for n in range(1, len(questionnaires) + 1):
        for combo in itertools.combinations(questionnaires, n):
            combo_names = [q[0] for q in combo]
            data_paths = [q[1] for q in combo]
            criterias = [q[2] for q in combo]

            # Load and preprocess data
            X, y = load_and_preprocess(data_paths, criterias, diagnosis_path, balance=True)
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Train the model
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)

            # Evaluate the model
            y_pred = clf.predict(X_test)
            print(f"Model trained with: {combo_names}")
            print(y.value_counts())
            print(classification_report(y_test, y_pred))

            # Save the model
            model_filename = 'saved_models/'+'rf_' + '_'.join(combo_names) + '.pkl'
            joblib.dump((clf, X.columns.tolist()), model_filename)