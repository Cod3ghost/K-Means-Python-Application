"""
preprocessing.py

A minimal, dependency-free (numpy-only) StandardScaler, written from scratch
because scikit-learn cannot be installed in this project's execution
environment. Behaves the same way as sklearn.preprocessing.StandardScaler
for the purposes of this application: it centers each feature to zero mean
and scales it to unit variance (population standard deviation, ddof=0).

Author: Desmond Amos Bature
Course: BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment - K-Means Python Application
"""

import numpy as np


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per-feature mean learned during fit.
    scale_ : ndarray of shape (n_features,)
        Per-feature standard deviation learned during fit. Features with a
        standard deviation of zero (constant columns) are left un-scaled
        (scale_ is set to 1.0 for those columns) to avoid division by zero.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        # Guard against zero-variance (constant) columns.
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler instance is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
