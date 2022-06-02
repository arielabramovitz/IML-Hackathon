import pandas as pd
import numpy as np
from datetime import datetime


def make_column_timestamp(data_frame, col_name):
    data_frame[col_name] = \
        data_frame[col_name].astype(str).str[0:10]

    data_frame[col_name] = \
        data_frame[col_name].replace('Unknown', np.NaN)

    data_frame[col_name] = \
        data_frame[col_name].astype('datetime64[ns]')

    data_frame[col_name] = data_frame[col_name].values.astype(np.int64) // 10 ** 9


def parse():
    # Use a breakpoint in the code line below to debug your script.
    data_frame = pd.read_csv('data/train.feats.csv')

    make_column_timestamp(data_frame, 'אבחנה-Diagnosis date')
    make_column_timestamp(data_frame, 'surgery before or after-Activity date')
    make_column_timestamp(data_frame, 'אבחנה-Surgery date1')

    data_frame.drop(columns=[' Hospital', 'User Name', "id-hushed_internalpatientid"], inplace=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Basic stage'])
    data_frame = pd.get_dummies(data_frame, columns=["אבחנה-Histopatological degree"], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=["אבחנה-Histological diagnosis"], drop_first=True)
    data_frame = data_frame["אבחנה-Positive nodes"].fillna(0.0)
    # TODO Her2 - check unique and make the same

    print(data_frame)


if __name__ == '__main__':
    parse()
