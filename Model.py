import typing
import Data.evaluate_part_0 as evaluate_file
import os
from subprocess import Popen, PIPE
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
import optuna
import torch
import shlex
import streamlit

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import plotly.graph_objs as go

# ___ Magic Numbers___

# Data Split:
TEST_SIZE = 0.15
TRAIN_SIZE = 0.85

GOLD_FILE = "Data/train.labels.0.csv"
EVALUATION_FILE = "Data/evaluate_part_0.py"


def split_data(dt: DataFrame):
    """
    Splitting the Data Frame to Train and Test
    """
    train, test = train_test_split(dt, test_size=TEST_SIZE,
                                   train_size=TRAIN_SIZE)
    return train, test


def compress_data(X: DataFrame, y: DataFrame):
    pass


def find_optimal_hyperparameter(X: DataFrame, y: Series, estimator: BaseEstimator, param_grid: dict, cv: int):
    grid_cv = GridSearchCV(estimator, param_grid, cv=cv)
    grid_cv.fit(X, y)
    return grid_cv.cv_results_

def objective(trial):
    """
    Evaluates the micro and the macro, I will need to choose by hand to which
    one I care
    """
    macro_f1, micro_f1 = evaluate_file.evaluate(GOLD_FILE, GOLD_FILE)

    return macro_f1


def learners_list():
    learners = list()


if __name__ == "__main__":
    np.random.seed(0)

