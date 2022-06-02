import pandas as pd
import numpy as np
from datetime import datetime

def make_column_timestamp(data_frame, col_name):
    data_frame[col_name] =\
        data_frame[col_name].astype(str).str[0:10]

    data_frame[col_name] =\
        data_frame[col_name].replace('Unknown', np.NaN)

    data_frame[col_name] =\
        data_frame[col_name].astype('datetime64[ns]')

    data_frame[col_name] = data_frame[col_name].values.astype(np.int64) // 10 ** 9

def parse():
    # Use a breakpoint in the code line below to debug your script.
    data_frame = pd.read_csv('train.feats.csv')

    data_frame.drop(
        columns=[
            ' Form Name',
            ' Hospital',
            'User Name',
            'id-hushed_internalpatientid',
            'surgery before or after-Activity date',
            'אבחנה-Tumor depth',
            'אבחנה-Tumor width',
            'אבחנה-Surgery date1',
            'אבחנה-Surgery date2',
            'אבחנה-Surgery date3',
            'אבחנה-Surgery name3'
        ], inplace=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Basic stage'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-T -Tumor mark (TNM)'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['surgery before or after-Actual activity'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Surgery name1'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Surgery name2'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-M -metastases mark (TNM)'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Margin Type'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Lymphatic penetration'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Side'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Histological diagnosis'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Histopatological degree'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Surgery sum'], drop_first=True)

    make_column_timestamp(data_frame, 'אבחנה-Diagnosis date')

    data_frame["decade_born"] = (data_frame["אבחנה-Age"]/10).astype(int)
    data_frame = pd.get_dummies(data_frame, columns=["decade_born"], drop_first=True)
    data_frame = data_frame.drop(["אבחנה-Age"], 1)
    #data_frame = data_frame.drop("yr_built", 1)
    print()

    #make_column_timestamp(data_frame, 'אבחנה-Surgery date2')

   # print(data_frame['אבחנה-Surgery sum'].value_counts())
    #print(data_frame['אבחנה-Margin Type'].value_counts())


if __name__ == '__main__':
    parse()