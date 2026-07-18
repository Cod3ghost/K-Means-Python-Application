"""
metrics.py

Cluster evaluation metrics implemented from scratch in numpy (silhouette
score) plus simple label-agreement diagnostics (purity and a cross-tab)
used only to sanity-check the unsupervised clusters against the known
cancer-type labels that ship with the TCGA-PANCAN dataset. The clustering
itself never sees these labels; they are used solely for post-hoc
evaluation, consistent with an unsupervised learning workflow.

Author: Desmond Amos Bature
Course: BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment - K-Means Python Application
"""

import numpy as np
import pandas as pd


def _pairwise_euclidean(X):
    X_sq = np.sum(X ** 2, axis=1)
    dists = X_sq[:, None] + X_sq[None, :] - 2 * (X @ X.T)
    np.fill_diagonal(dists, 0.0)
    dists = np.maximum(dists, 0.0)
    return np.sqrt(dists)


def silhouette_score(X, labels, sample_size=None, random_state=None):
    """Mean silhouette coefficient over all samples.

    For each point i: a(i) = mean distance to other points in its own
    cluster; b(i) = mean distance to points in the nearest other cluster;
    s(i) = (b(i) - a(i)) / max(a(i), b(i)). Returns the mean of s(i).

    O(n^2) in the number of rows passed in, which is fine for the sample
    sizes used in this project (a few hundred to a couple thousand rows).
    For larger inputs, pass sample_size to evaluate on a random subset.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X = X[idx]
        labels = labels[idx]

    n = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Silhouette score requires at least 2 clusters.")

    dist_matrix = _pairwise_euclidean(X)
    scores = np.zeros(n, dtype=np.float64)

    for label in unique_labels:
        in_cluster = labels == label
        cluster_idx = np.where(in_cluster)[0]
        cluster_size = len(cluster_idx)

        for i in cluster_idx:
            if cluster_size > 1:
                a_i = dist_matrix[i, cluster_idx].sum() / (cluster_size - 1)
            else:
                a_i = 0.0

            b_i = np.inf
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other_idx = np.where(labels == other_label)[0]
                mean_dist = dist_matrix[i, other_idx].mean()
                b_i = min(b_i, mean_dist)

            scores[i] = 0.0 if max(a_i, b_i) == 0 else (b_i - a_i) / max(a_i, b_i)

    return float(scores.mean())


def cluster_purity(pred_labels, true_labels):
    """Fraction of samples assigned to the majority true class within their
    predicted cluster (post-hoc validation metric, not used during fitting).
    """
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    n = len(pred_labels)
    correct = 0
    for cluster_id in np.unique(pred_labels):
        mask = pred_labels == cluster_id
        if mask.sum() == 0:
            continue
        values, counts = np.unique(true_labels[mask], return_counts=True)
        correct += counts.max()
    return correct / n


def cross_tab(pred_labels, true_labels, pred_name="Cluster", true_name="Cancer Type"):
    """Contingency table of predicted cluster vs. known cancer type."""
    df = pd.DataFrame({pred_name: pred_labels, true_name: true_labels})
    return pd.crosstab(df[pred_name], df[true_name])
