import os
import sys
import pandas as pd
import numpy as np
import csv
import Clean
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import Data.evaluate_part_0 as evaluate_file_0
import Data.evaluate_part_1 as evaluate_file_1

ind_to_label = dict()
label_to_ind = dict()

TRAIN_LABELS_FN = "Data/train.labels.1.csv"
TRAIN_X_FN = "Data/train.feats.csv"


def create_multi_hot_labels(labels):
    """
    Turns the string labels into multi hot/
    """
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

def validate_1(y, X):
    # y = create_multi_hot_labels(y)
    # todo make the Tumor location thing into a multi hot feature too

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    tree = RandomForestRegressor()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_val)

    print("Evaluation: ", send_to_evaluation_1(y_val, y_pred))


def send_to_evaluation_1(y_gold, y_pred):
    """ Receives our multi-hot, puts it in a csv and evaluates"""
    y_gold_df = pd.DataFrame(list(y_gold))
    y_pred_df = pd.DataFrame(list(y_pred))
    y_gold_df.to_csv('temp_gold.labels.1.csv', index=False)
    y_pred_df.to_csv('temp_pred.labels.1.csv', index=False)
    mse = evaluate_file_1.evaluate('temp_gold.labels.1.csv',
                                                "temp_pred.labels.1.csv")
    os.remove('temp_gold.labels.1.csv')
    os.remove('temp_pred.labels.1.csv')
    return mse


def predict():
    pass

def fit():
    pass

if __name__ == '__main__':
    np.random.seed(0)
    # todo change this all to train on all the data and create a prediction
    df_fn, labels_fn = sys.argv[1], sys.argv[2]
    df = Clean.parse(df_fn, labels_fn)
    y_1, X_1 = df["אבחנה-Tumor size"], df.drop(
        columns=["אבחנה-Tumor size"])


    validate_1(y_1, X_1)
