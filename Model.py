# import typing
from sklearn.ensemble import RandomForestClassifier

import Data.evaluate_part_0 as evaluate_file
# import os
# from subprocess import Popen, PIPE
from sklearn.base import BaseEstimator
from sklearn import ensemble
# from sklearn.cluster import SpectralClustering, KMeans
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LassoCV
import sklearn.multiclass
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
import optuna
# import torch
# import shlex
# import streamlit

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
# import plotly.graph_objs as go

# ___ Magic Numbers___

# Data Split:
TEST_SIZE = 0.15
TRAIN_SIZE = 0.85

GOLD_FILE = "Data/train.labels.0.csv"
EVALUATION_FILE = "Data/evaluate_part_0.py"


def take_data():
    """ I am just trying to take a single numeric feature and try to study it"""
    df = pd.read_csv("Data/train.feats.csv")
    X = df["אבחנה-Age"]



def find_optimal_hyperparameter(X: DataFrame, y: Series, estimator: BaseEstimator, param_grid: dict, cv: int):
    grid_cv = GridSearchCV(estimator, param_grid, cv=cv)
    grid_cv.fit(X, y)
    return grid_cv.cv_results_

def objective(trial):
    """
    Evaluates the micro and the macro, I will need to choose by hand to which
    one I care
    """

    rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
    classifier_obj = ensemble.RandomForestClassifier(
        max_depth=rf_max_depth, n_estimators=rf_n_estimators)


    macro_f1, micro_f1 = evaluate_file.evaluate(GOLD_FILE, GOLD_FILE)
    return macro_f1
    # x = trial.suggest_float("x", -10, 10)
    # return (x - 2) ** 2


def choose_model(X_train, y_train):

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    return best_params



if __name__ == "__main__":
    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tree = RandomForestClassifier()
    tree.fit(X_train, y_train)
    pred_y = tree.predict(X_test)

