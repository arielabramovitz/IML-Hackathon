from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA

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

def optimize_logistic_regression():
    pass

def optimize_nearest_neighbor():
    pass

def optimize_SVM():
    pass

def optimize_kernel_SVM():
    pass

def optimize_naive_base():
    pass

def optimize_decision_tree():
    pass

def optimize_random_forest():
    pass


if __name__ == "__main__":
    np.random.seed(0)
    pass