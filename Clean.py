import pandas as pd

def parse():
    # Use a breakpoint in the code line below to debug your script.
    data_frame = pd.read_csv(
        'train.feats.csv',
        parse_dates=["אבחנה-Diagnosis date", "surgery before or after-Activity date"])

    data_frame.drop(columns=[' Hospital', 'User Name', "id-hushed_internalpatientid"], inplace=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Basic stage'])


    # TODO Her2 - check unique and make the same

    print(data_frame.head())

if __name__ == '__main__':
    parse()