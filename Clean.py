import pandas as pd
import numpy as np


def her2_column(data, col_name):
    neg_regex = '[NnGg]|[של]|0'
    pos_regex = '[PpSs]|[חב]|[123]|-|\+'
    data[col_name].replace(pos_regex, 1, regex=True, inplace=True)
    data[col_name].replace(neg_regex, 0, regex=True, inplace=True)
    data[col_name].fillna(0, inplace=True)
    temp = (data[col_name] == 0) | (data[col_name] == 1)
    data = data[temp]
    return data


def make_column_timestamp(data_frame, col_name):
    data_frame[col_name] = \
        data_frame[col_name].astype(str).str[0:10]

    data_frame[col_name] = \
        data_frame[col_name].replace('Unknown', np.NaN)

    data_frame[col_name] = \
        data_frame[col_name].astype('datetime64[ns]')

    data_frame[col_name] = data_frame[col_name].values.astype(np.int64) // 10 ** 9


def parse(X_df_fn, y_df_fn=None, y_df_fn_2=None):

    # Use a breakpoint in the code line below to debug your script.
    data_frame = pd.read_csv(X_df_fn)

    # Can do the clean for X, y, and another y
    if y_df_fn != None:
        labels = pd.read_csv(y_df_fn)
        # This is done so that the rows that are removed will be for the labels too
        data_frame = pd.concat(objs=[labels, data_frame], axis=1)

    if y_df_fn_2 != None:
        labels = pd.read_csv(y_df_fn_2)
        # This is done so that the rows that are removed will be for the labels too
        data_frame = pd.concat(objs=[labels, data_frame], axis=1)

    data_frame.drop(
        columns=[
            ' Form Name',
            ' Hospital',
            'User Name',
            'אבחנה-Ivi -Lymphovascular invasion',
            'אבחנה-KI67 protein',
            'אבחנה-Side',
            'אבחנה-Stage',

            'אבחנה-Surgery date1',
            'אבחנה-Surgery name1',
            'אבחנה-Surgery date2',
            'אבחנה-Surgery name2',
            'אבחנה-Surgery date3',
            'אבחנה-Surgery name3',

            'אבחנה-Tumor depth',
            'אבחנה-Tumor width',
            'אבחנה-Surgery sum',

            'אבחנה-er',
            'אבחנה-pr',

            'surgery before or after-Activity date',
            'surgery before or after-Actual activity',
            'id-hushed_internalpatientid',

            'אבחנה-N -lymph nodes mark (TNM)'
        ], inplace=True)


    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Basic stage'], drop_first=True)
    data_frame = her2_column(data_frame, 'אבחנה-Her2')
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Her2'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Histological diagnosis'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Histopatological degree'], drop_first=True)

    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Lymphatic penetration'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-M -metastases mark (TNM)'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Margin Type'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-T -Tumor mark (TNM)'], drop_first=True)

    data_frame['אבחנה-Nodes exam'] = data_frame['אבחנה-Nodes exam'].fillna(0)
    data_frame['אבחנה-Positive nodes'] = data_frame['אבחנה-Positive nodes'].fillna(0)


    make_column_timestamp(data_frame,'אבחנה-Diagnosis date')

    return data_frame
