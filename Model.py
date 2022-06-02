import typing

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, train_test_split, KFold

import streamlit

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import plotly.graph_objs as go

# ___ Magic Numbers___

# Data Split:
TEST_SIZE = 0.15
TRAIN_SIZE = 0.85


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

# ------------------------------------------------------ #

def optimize_logistic_regression():
    """ Hyper-parameter optimization"""
    pass


def optimize_nearest_neighbor():
    """ Hyper-parameter optimization"""
    pass


def optimize_SVM():
    """ Hyper-parameter optimization"""
    pass


def optimize_kernel_SVM():
    """ Hyper-parameter optimization"""
    pass


def optimize_naive_base():
    """ Hyper-parameter optimization"""
    pass


def optimize_decision_tree():
    """ Hyper-parameter optimization"""
    pass


def optimize_random_forest():
    """ Hyper-parameter optimization"""
    pass


# ------------------------------------------------------ #

def compare_learners(learner_list: list):
    """ Receiving a list of different learners, trains them and compares their
     success.
     returns the most successful one."""
    # todo draw a nice graph
    pass


if __name__ == "__main__":
    np.random.seed(0)
    pass
