""" 
Author: George L. Roberts
Date: 28-03-2020
About: Analysis of kaggle credit card fraud dataset
https://www.kaggle.com/mlg-ulb/creditcardfraud
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')
PLOT_DIR = os.path.join(CDIR, 'plots')


def main():
    data = load_data()
    eda = ExploratoryDataAnalysis(data)
    # eda.print_all()
    eda.plot_all()


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'creditcard.csv'))


class ExploratoryDataAnalysis():
    """ Initial dataset exploration
    Observations:
        No nans in the dataset.
        Highly imbalanced (around 500 times fewer anomalies)
        Reducing any dimensions strongly reduces the explained
            variance.
    TODO: UMAP and view clusters, see if outliers immediately obvious
    TODO: Plot a heatmap to show feature correlation
    TODO: Think of how to incorporate time feature.
    """
    def __init__(self, data):
        self.data = data

    def print_all(self):
        """ Do everything that requires printing to console """
        print(self.data.head())
        print(self.data.columns)
        print(self.data.describe())
        self.count_nans()
        self.class_imbalance()

    def plot_all(self):
        self.plot_PCA()
        self.plot_explained_variance()

    def count_nans(self):
        """ Count nans in dataframe columns """
        any_nans = 0
        for column in self.data.columns:
            col_nans = np.sum(np.isnan(self.data[column]))
            if col_nans > 0:
                print(f'{column}: {col_nans} nans')
            any_nans += col_nans
        if any_nans == 0:
            print('No nans found!')

    def class_imbalance(self):
        """ Show difference between anomalies and correct data """
        labels = self.data['Class']
        anomalies = np.sum(labels)
        real = len(labels) - anomalies
        proportion = real / anomalies
        print(f'{real} real, {anomalies} anomalies:')
        print(f'{proportion:.2f} times more real')

    def plot_PCA(self):
        """ Principal component analysis to identify any patterns
        """
        features, labels = self.get_feats_labels()
        features = StandardScaler().fit_transform(features)

        pca = PCA(n_components=2)
        components = pca.fit_transform(features)
        real = components[labels == 0]
        anomalies = components[labels==1]
        fig, ax = plt.subplots()
        ax.scatter(real[:, 0], real[:, 1], color='g', label='real')
        ax.scatter(anomalies[:, 0], anomalies[:, 1], color='r', label='anomaly')
        ax.set_xlabel('Principal component 1')
        ax.set_ylabel('Principal component 2')
        ax.legend()

        plt.tight_layout()
        fpath = os.path.join(PLOT_DIR, 'pca_2.png')
        plt.savefig(fpath)
        plt.close()

    def plot_explained_variance(self):
        features, _ = self.get_feats_labels()
        features = StandardScaler().fit_transform(features)

        pca = PCA().fit(features)

        fig, ax = plt.subplots()
        ax.plot(np.cumsum(pca.explained_variance_ratio_))
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Variance (%)') #for each component

        plt.tight_layout()
        fpath = os.path.join(PLOT_DIR, 'explained_variance.png')
        plt.savefig(fpath)
        plt.close()
        breakpoint()

    def get_feats_labels(self):
        not_feats = ['Time', 'Class']
        feature_cols = [x for x in self.data.columns if x not in not_feats]
        features = self.data[feature_cols].values
        labels = self.data['Class'].values
        return features, labels

if __name__ == "__main__":
    main()
