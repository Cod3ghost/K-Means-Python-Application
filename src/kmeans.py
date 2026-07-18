"""
kmeans.py

A from-scratch implementation of the K-Means clustering algorithm (Lloyd's
algorithm with k-means++ centroid initialization), written in pure numpy.
scikit-learn could not be installed in this project's execution environment
(no package-index access from the sandbox used to build this application),
so the core algorithm required by the assignment is implemented directly
rather than imported. This also demonstrates the mechanics behind the
"assign to nearest centroid, recompute centroid, repeat" loop described in
Module 4 (Clustering and Dimensionality Reduction).

Author: Desmond Amos Bature
Course: BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment - K-Means Python Application
"""

import numpy as np


def _pairwise_sq_dists(X, centers):
    """Squared Euclidean distance between every row of X and every center."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
    X_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]
    C_sq = np.sum(centers ** 2, axis=1)[np.newaxis, :]
    cross = X @ centers.T
    dists = X_sq - 2 * cross + C_sq
    return np.maximum(dists, 0.0)


class KMeans:
    """K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters, k.
    max_iter : int
        Maximum number of Lloyd iterations per run.
    n_init : int
        Number of independent random (k-means++) initializations to try.
        The run with the lowest final inertia is kept, which reduces the
        chance of converging to a poor local optimum.
    tol : float
        Convergence tolerance on the shift in centroid positions between
        iterations (Frobenius norm).
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    labels_ : ndarray of shape (n_samples,)
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center
        (within-cluster sum of squares).
    n_iter_ : int
        Number of iterations run in the best initialization.
    """

    def __init__(self, n_clusters=3, max_iter=300, n_init=10, tol=1e-4, random_state=None):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _kmeans_plus_plus_init(self, X, rng):
        n_samples = X.shape[0]
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)

        first_idx = rng.integers(n_samples)
        centers[0] = X[first_idx]

        closest_sq_dist = _pairwise_sq_dists(X, centers[0:1]).ravel()

        for c in range(1, self.n_clusters):
            probs = closest_sq_dist / closest_sq_dist.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centers[c] = X[next_idx]
            new_sq_dist = _pairwise_sq_dists(X, centers[c:c + 1]).ravel()
            closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

        return centers

    def _single_run(self, X, rng):
        n_samples = X.shape[0]
        centers = self._kmeans_plus_plus_init(X, rng)
        labels = np.zeros(n_samples, dtype=np.int64)

        for iteration in range(1, self.max_iter + 1):
            dists = _pairwise_sq_dists(X, centers)
            new_labels = np.argmin(dists, axis=1)

            new_centers = centers.copy()
            for k in range(self.n_clusters):
                mask = new_labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    # Re-seed an empty cluster on the point farthest from its
                    # own centroid, so no cluster silently disappears.
                    farthest_idx = np.argmax(np.min(dists, axis=1))
                    new_centers[k] = X[farthest_idx]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            labels = new_labels

            if shift <= self.tol:
                break

        final_dists = _pairwise_sq_dists(X, centers)
        inertia = float(np.sum(np.min(final_dists, axis=1)))
        return centers, labels, inertia, iteration

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        rng = np.random.default_rng(self.random_state)

        best_inertia = None
        best_centers = None
        best_labels = None
        best_n_iter = None

        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X, rng)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans instance is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        dists = _pairwise_sq_dists(X, self.cluster_centers_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
