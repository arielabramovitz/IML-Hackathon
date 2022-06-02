import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import Data.evaluate_part_0 as evaluate_file

ind_to_label = dict()
label_to_ind = dict()

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

def create_string_labeled_data(predictions):
    converted = []
    for line in predictions:
        curr = []
        for i in range(len(line)):
            if line[i] == 1:
                curr.append(ind_to_label[i])
        converted.append(str(curr))
    converted
    return converted


def create_multi_hot_labels(labels):
    classes = []
    data = []
    for row in labels:
        s = str(row).strip("[").strip("]").replace("'", "").split(", ")
        data.append(s)
        for i in s:
            if len(i) > 0 and i not in classes:
                ind_to_label[len(classes)] = i
                label_to_ind[i] = len(classes)
                classes.append(i)
    m = MultiLabelBinarizer(classes=classes)
    m.fit(data)
    return m.transform(data)


def parse():
    # Use a breakpoint in the code line below to debug your script.
    data_frame = pd.read_csv('data/train.feats.csv')
    labels = pd.read_csv("data/train.labels.0.csv")
    data_frame = pd.concat(objs=[labels, data_frame], axis=1)

    data_frame.drop(
        columns=[
            ' Form Name',
            ' Hospital',
            'User Name',
            'אבחנה-Diagnosis date',
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

    data_frame["decade_born"] = (data_frame["אבחנה-Age"] / 10).astype(int)
    data_frame = pd.get_dummies(data_frame, columns=["decade_born"], drop_first=True)
    data_frame.drop(['אבחנה-Age'], inplace=True, axis=1)

    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Basic stage'], drop_first=True)
    data_frame = her2_column(data_frame, 'אבחנה-Her2')
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Her2'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Histological diagnosis'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Histopatological degree'], drop_first=True)

    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Lymphatic penetration'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-M -metastases mark (TNM)'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-Margin Type'], drop_first=True)
    data_frame = pd.get_dummies(data_frame, columns=['אבחנה-T -Tumor mark (TNM)'], drop_first=True)

    data_frame["nodes_exam_pref"] = (data_frame['אבחנה-Nodes exam'].fillna(0) // 10).astype(int)
    data_frame.drop(['אבחנה-Nodes exam'], inplace=True, axis=1)

    data_frame["pos_nodes_pref"] = (data_frame['אבחנה-Positive nodes'].fillna(0) // 10).astype(int)
    data_frame.drop(['אבחנה-Positive nodes'], inplace=True, axis=1)

    y, X = data_frame["אבחנה-Location of distal metastases"], data_frame.drop(
        columns=["אבחנה-Location of distal metastases"])

    y = create_multi_hot_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tree = RandomForestClassifier()
    tree.fit(X_train, y_train)
    # score = tree.score(X_test, y_test)
    pred_y = tree.predict(X_test)
    y_pred_lst = create_string_labeled_data(pred_y)
    print(y_pred_lst)
    y_test_lst = create_string_labeled_data(y_test)
    print(y_test_lst)
    df_pred = pd.DataFrame(y_pred_lst)
    df_test = pd.DataFrame(y_test_lst)
    df_pred.to_csv('pred.labels.0.csv', index=False)
    df_test.to_csv('test.labels.0.csv', index=False)
    print(len(df_pred.columns))
    print(df_test)
    macro_f1, micro_f1 = evaluate_file.evaluate("pred.labels.0.csv", 'test.labels.0.csv')

if __name__ == '__main__':
    parse()
