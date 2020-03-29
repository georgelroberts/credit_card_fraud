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
import seaborn as sns
import umap
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
sns.set()

CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')
PLOT_DIR = os.path.join(CDIR, 'plots')


def main():
    data = load_data()
    eda = ExploratoryDataAnalysis(data, print_out=False, plot_out=False)
    modelling = Modelling(data, tuning=False)


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'creditcard.csv'))


class Modelling():
    """ Model the dataset
    TODO: Use SMOTE/lightgbm imbalance parameter
    TODO: Think carefully about which score to use. Show F1 and AUC first.
    """
    def __init__(self, data, tuning=False):
        self.data = data
        if tuning:
            self.hyperparameter_tuning()
        else:
            self.complete_model()

    def complete_model(self):
        train_X, test_X, train_y, test_y = self.preprocess_data()
        clf = lgb.LGBMClassifier()
        clf.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        pred_f1 = f1_score(test_y, pred_y)
        pred_auc = roc_auc_score(test_y, pred_y)
        print(f'Test F1: {pred_f1}; Test ROC-AUC: {pred_auc}')

    def preprocess_data(self, upsample=False):
        """ Separate into train and test 
        Use either upsampling or stratified splitting
        """
        cols = self.data.columns
        y_col = 'Class'
        X_cols = [x for x in cols if x != y_col]
        all_X = self.data[X_cols]
        all_y = self.data[y_col]
        if upsample:
            # Do SMOTE upsampling
            return
        else:
            train_X, test_X, train_y, test_y = train_test_split(
                    all_X, all_y, stratify=all_y, test_size=0.3)
            return train_X, test_X, train_y, test_y

    def hyperparameter_tuning(self):
        train_X, _, train_y, _ = self.preprocess_data()
        train_X, cv_X, train_y, cv_y = train_test_split(
                train_X, train_y, stratify=train_y, test_size=0.3)


class ExploratoryDataAnalysis():
    """ Initial dataset exploration
    Observations:
        No nans in the dataset.
        Highly imbalanced (around 500 times fewer anomalies)
        Reducing any dimensions strongly reduces the explained
            variance.
        Very little correlation between features (only small
            correlations with amount and other features)
        Looking at the pairplot, it actually looks as if the features
            do diverge a fair bit, F1 could be high for this
        UMAP doesn't help, clusters created don't help identify the
            classes
    """
    def __init__(self, data, print_out=True, plot_out=True):
        self.data = data
        if print_out:
            self.print_all()
        if plot_out:
            self.plot_all()

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
        self.plot_umap()
        self.plot_structured_heatmap()
        self.plot_pairplot()

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
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

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

    def plot_structured_heatmap(self):
        features, _ = self.get_feats_labels()

        sns.clustermap(features.corr(), center=0, cmap="vlag", linewidths=.75,
                figsize=(13, 13))
        plt.tight_layout()
        fpath = os.path.join(PLOT_DIR, 'structured_heatmap.png')
        plt.savefig(fpath)
        plt.close()

    def plot_pairplot(self):
        sns.pairplot(self.data[['Time', 'Amount', 'Class']], hue='Class')
        plt.tight_layout()
        fpath = os.path.join(PLOT_DIR, 'pair_plot_small.png')
        plt.savefig(fpath)
        plt.close()

        sns.pairplot(self.data, hue='Class')
        plt.tight_layout()
        fpath = os.path.join(PLOT_DIR, 'pair_plot.png')
        plt.savefig(fpath)
        plt.close()

    def plot_umap(self):
        """
        It's important to plot umap at different scales to identify
        different clusters
        """
        features, labels = self.get_feats_labels()
        no_samples = 50000
        features = features.sample(no_samples).values
        labels = labels.sample(no_samples).values
        neighbours = [3, 10, 100]
        min_dists = [0.1, 0.5, 0.99]
        for neighbour in neighbours:
            for min_dist in min_dists:
                reducer = umap.UMAP(n_neighbors=neighbour, min_dist=min_dist)
                embedding = reducer.fit_transform(features)
                fig, ax = plt.subplots()
                colours = [sns.color_palette()[x] for x in labels]
                ax.scatter(embedding[:, 0], embedding[:, 1], c=colours)
                min_dist_str = str(min_dist).replace('.', '-')
                plt.tight_layout()
                fname = f'umap_{neighbour}-neigbhours_{min_dist_str}-mindist.png'
                fpath = os.path.join(PLOT_DIR, fname)
                plt.savefig(fpath)
                plt.close()

    def get_feats_labels(self):
        label_col = 'Class'
        feature_cols = [x for x in self.data.columns if x != label_col] 
        features = self.data[feature_cols]
        labels = self.data[label_col]
        return features, labels

if __name__ == "__main__":
    main()
