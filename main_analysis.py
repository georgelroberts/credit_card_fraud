""" 
Author: George L. Roberts
Date: 28-03-2020
About: Analysis of kaggle credit card fraud dataset
https://www.kaggle.com/mlg-ulb/creditcardfraud
"""

import os
import pandas as pd
import numpy as np

CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')


def main():
    data = load_data()
    eda = ExploratoryDataAnalysis(data)
    eda.print_all()


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'creditcard.csv'))


class ExploratoryDataAnalysis():
    """ Initial dataset exploration
    TODO: UMAP and view clusters, see if outliers immediately obvious
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


if __name__ == "__main__":
    main()
