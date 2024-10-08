import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def get_mimic_dataLoader(data_dir='./datasets/parabola',
                        type='SGD', config=None,
                        batch_size=None):

    scaler = StandardScaler()

    # Load and preprocess the data
    df = pd.read_csv('./datasets/MIMIC/cohorta_trial1_data.csv')
    # Replace all missing values with 0
    df.fillna(0, inplace=True)
    df_sota = pd.read_csv('./datasets/MIMIC/cohorta_sfd.csv')
    df_sota.fillna(0, inplace=True)

    # Features
    X = df[['subject_id', 'hadm_id', 'stay_id', 'dod', 'mortality_year',
            'rsp_pao2fio2_vent_min', 'rsp_pao2fio2_novent_min',
            'cgn_platelet_min',
            'cdv_mbp_min', 'cdv_rate_dopamine', 'cdv_rate_dobutamine',
            'cdv_rate_epinephrine', 'cdv_rate_norepinephrine',
            'gcs_min', 'rfl_urineoutput', 'rfl_creatinine_max']]

    # join the df and df_sota based on subjec_id, hadm_id, stay_id and dod
    df_final = pd.merge(X, df_sota,
                        on=['subject_id', 'hadm_id', 'stay_id', 'mortality_year'],
                        how='inner')

    # make the following mapping for the concepts:
    df_final['respiration'] = df_final['respiration'].replace(1, 0)
    df_final['respiration'] = df_final['respiration'].replace(2, 1)
    df_final['respiration'] = df_final['respiration'].replace(3, 1)
    df_final['respiration'] = df_final['respiration'].replace(4, 2)
    df_final['renal'] = df_final['renal'].replace(1, 0)
    df_final['renal'] = df_final['renal'].replace(2, 1)
    df_final['renal'] = df_final['renal'].replace(3, 1)
    df_final['renal'] = df_final['renal'].replace(4, 2)
    df_final['coagulation'] = df_final['coagulation'].replace(1, 0)
    df_final['coagulation'] = df_final['coagulation'].replace(2, 1)
    df_final['coagulation'] = df_final['coagulation'].replace(3, 1)
    df_final['coagulation'] = df_final['coagulation'].replace(4, 2)
    df_final['liver'] = df_final['liver'].replace(1, 0)
    df_final['liver'] = df_final['liver'].replace(2, 1)
    df_final['liver'] = df_final['liver'].replace(3, 1)
    df_final['liver'] = df_final['liver'].replace(4, 2)
    df_final['cardiovascular'] = df_final['cardiovascular'].replace(1, 0)
    df_final['cardiovascular'] = df_final['cardiovascular'].replace(2, 1)
    df_final['cardiovascular'] = df_final['cardiovascular'].replace(3, 1)
    df_final['cardiovascular'] = df_final['cardiovascular'].replace(4, 2)
    df_final['cns'] = df_final['cns'].replace(1, 0)
    df_final['cns'] = df_final['cns'].replace(2, 1)
    df_final['cns'] = df_final['cns'].replace(3, 1)
    df_final['cns'] = df_final['cns'].replace(4, 2)

    # Features
    X = df_final[
        ['rsp_pao2fio2_vent_min', 'rsp_pao2fio2_novent_min', 'cgn_platelet_min',
         'cdv_mbp_min', 'cdv_rate_dopamine', 'cdv_rate_dobutamine',
         'cdv_rate_epinephrine', 'cdv_rate_norepinephrine',
         'gcs_min', 'rfl_urineoutput', 'rfl_creatinine_max']]

    concept_names = {
        'respiration': ['normal', 'moderate', 'severe'],
        'renal': ['normal', 'moderate', 'severe'],
        'coagulation': ['normal', 'moderate', 'severe'],
        'liver': ['normal', 'moderate', 'severe'],
        'cardiovascular': ['normal', 'moderate', 'severe'],
        'cns': ['normal', 'moderate', 'severe']
    }
    num_concepts = 18
    num_classes = 1

    # Concepts
    C = df_final[['respiration', 'renal', 'coagulation',
                  'liver', 'cardiovascular','cns']]

    # convert the concepts to one-hot encoding
    # Fit and transform the matrix C to one-hot encoded format
    encoder = OneHotEncoder(
        sparse_output=False)  # Set sparse_output to False to get a dense matrix
    C = encoder.fit_transform(C)

    # Label
    Y = df['mortality_year']

    # Convert to tensors
    X_train = torch.tensor(X.values, dtype=torch.float32)
    C_train= torch.tensor(C, dtype=torch.float32)
    y_train = torch.tensor(Y.values, dtype=torch.long)

    # Normalize the features
    X_scaled = scaler.fit_transform(X)
    X_tensor_scaled = torch.tensor(X_scaled, dtype=torch.float32)

    # Split the data into temporary (80%) and test (20%) sets
    X_train, X_val, C_train, C_val, y_train, y_val = train_test_split(
        X_tensor_scaled, C_train, y_train, test_size=0.20, random_state=42)

    # Further split the temporary data into training (75% of temporary) and validation (25% of temporary) sets
    X_val, X_test, C_val, C_test, y_val, y_test = train_test_split(
        X_val, C_val, y_val, test_size=0.25, random_state=42)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, C_train, y_train)
    val_dataset = TensorDataset(X_val, C_val, y_val)
    test_dataset = TensorDataset(X_test, C_test, y_test)

    if type == 'SGD':
        data_train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        data_val_loader = DataLoader(dataset=val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
        data_test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
    else:
        NotImplementedError('ERROR: data type not supported!')

    return data_train_loader, data_val_loader, data_test_loader