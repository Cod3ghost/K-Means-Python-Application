"""
pca.py

A from-scratch implementation of Principal Component Analysis (PCA) using
numpy's singular value decomposition (SVD). Written without scikit-learn
because scikit-learn could not be installed in this project's execution
environment (no package-index access). This mirrors the module's coverage
of dimensionality reduction (Module 4: Clustering and Dimensionality
Reduction) and is used ahead of K-Means to compress the 20,531-gene
expression feature space down to a small number of components that retain
most of the variance in the data, before clustering.

Author: Desmond Amos Bature
Course: BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment - K-Means Python Application
"""

import numpy as np


class PCA:
    """Principal Component Analysis via SVD.

    Parameters
    ----------
    n_components : int or float, optional
        Number of components to keep. If a float in (0, 1], it is treated
        as the minimum proportion of cumulative explained variance to
        retain, and the number of components is chosen automatically.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, ordered by explained variance.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each selected component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each selected component.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean subtracted before projecting.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Economy-size SVD: X_centered = U @ diag(S) @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance

        if self.n_components is None:
            n_comp = Vt.shape[0]
        elif isinstance(self.n_components, float) and 0.0 < self.n_components <= 1.0:
            cumulative = np.cumsum(explained_variance_ratio)
            n_comp = int(np.searchsorted(cumulative, self.n_components) + 1)
        else:
            n_comp = int(self.n_components)

        self.components_ = Vt[:n_comp]
        self.explained_variance_ = explained_variance[:n_comp]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_comp]
        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
