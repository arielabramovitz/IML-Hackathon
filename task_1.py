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


def predict_to_file(tree, X_test):
    """Predicting the test and putting it in the file predictions.csv"""
    y_pred = tree.predict(X_test)
    y_pred_df = pd.DataFrame(list(y_pred))
    y_pred_df.to_csv('part1/predictions.csv', index=False)

def fit(X_train, y_train):
    """ Receives the X and y to train and returns a trained model"""
    tree = RandomForestRegressor()
    tree.fit(X_train, y_train)
    return tree

if __name__ == '__main__':
    np.random.seed(0)
    # todo change this all to train on all the data and create a prediction
    df_fn, labels_fn = sys.argv[1], sys.argv[2]
    df = Clean.parse(df_fn, labels_fn)
    y_1, X_1 = df["אבחנה-Tumor size"], df.drop(
        columns=["אבחנה-Tumor size"])


    validate_1(y_1, X_1)
