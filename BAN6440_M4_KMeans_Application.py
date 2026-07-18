"""
BAN6440_M4_KMeans_Application.py

K-Means Python Application - single-file submission.

Course:  BAN6440 - Applied Machine Learning for Business Analytics
Module:  Module 4 Assignment - K-Means Python Application
Author:  Desmond Amos Bature

WHAT THIS PROGRAM DOES
-----------------------
This program applies unsupervised K-Means clustering to gene expression
data from the National Cancer Institute Center for Cancer Research's
diffuse large B-cell lymphoma (DLBCL) project (NCICCR-DLBCL), registered
on the AWS Registry of Open Data and served by the Genomic Data Commons
(GDC). Without using any diagnosis or subtype label during fitting, it
tests whether gene expression alone separates a cohort of tumor samples
into distinct groups, then characterizes those groups after the fact
using a small panel of genes with known roles in B-cell lymphoma biology
(Alizadeh et al., 2000; Schmitz et al., 2018).

WHY EVERYTHING IS IMPLEMENTED FROM SCRATCH
--------------------------------------------
scikit-learn could not be installed in the sandboxed environment used to
develop this application (no package-index access, no root/administrator
access to install system packages, and no way to reach PyPI). Rather than
leave the assignment blocked on that, K-Means, PCA, a standard scaler, and
the evaluation metrics scikit-learn would normally provide are implemented
here directly in numpy, following the same well-known algorithms
scikit-learn itself uses: k-means++ initialization and Lloyd's algorithm
for K-Means, and singular value decomposition (SVD) for PCA. This is
validated by a companion unit test suite (test_kmeans_app.py, 22 tests,
all passing; see unit_test_results.log) that does not depend on
scikit-learn either.

HOW TO GET THE DATA AND RUN THIS
----------------------------------
1. Run download_dlbcl_data.py first (a separate, small helper script that
   ships alongside this file). It queries the GDC API directly over HTTPS
   and downloads the open-access, per-sample RNA-Seq gene count files for
   project NCICCR-DLBCL, no AWS account, AWS CLI, or authentication
   needed. It writes data/manifest.csv and data/raw/<file_id>.tsv.
2. Run this file: `python BAN6440_M4_KMeans_Application.py --data-dir data
   --output-dir output`. It loads the downloaded files, runs the full
   pipeline described below, and writes plots and a results summary to
   the output/ folder.

PIPELINE
--------
1. Load and assemble the DLBCL gene expression matrix from the per-sample
   files (Section 5, load_dlbcl_gene_expression).
2. Standardize features to zero mean and unit variance (Section 1,
   StandardScaler).
3. Reduce dimensionality with PCA to the number of components that
   explain 90% of variance (Section 2, PCA), both to counter the curse of
   dimensionality inherent to a several-thousand-gene feature space and
   because Module 4 pairs clustering with dimensionality reduction.
4. Use the elbow method (inertia) and the silhouette score to choose a
   defensible number of clusters, k, then fit the final K-Means model
   (Section 3, KMeans; Section 4, silhouette_score).
5. Characterize each cluster by its mean expression of curated DLBCL
   marker genes, a post-hoc interpretation step; no ground-truth subtype
   label is used at any point during fitting, since this dataset's
   open-access tier does not ship one (Section 6, run_pipeline).
6. Save the elbow plot, silhouette plot, a PCA cluster scatter plot, a
   results.json summary, and a cluster-assignment CSV to output/.

REFERENCES (NXU / APA style)
------------------------------
Alizadeh, A. A., Eisen, M. B., Davis, R. E., Ma, C., Lossos, I. S.,
    Rosenwald, A., Boldrick, J. C., Sabet, H., Tran, T., Yu, X., Powell,
    J. I., Yang, L., Marti, G. E., Moore, T., Hudson, J., Lu, L., Lewis,
    D. B., Tibshirani, R., Sherlock, G., ... Staudt, L. M. (2000).
    Distinct types of diffuse large B-cell lymphoma identified by gene
    expression profiling. Nature, 403(6769), 503 to 511.
Conesa, A., Madrigal, P., Tarazona, S., Gomez-Cabrero, D., Cervera, A.,
    McPherson, A., Szczesniak, M. W., Gaffney, D. J., Elo, L. L., Zhang,
    X., & Mortazavi, A. (2016). A survey of best practices for RNA-seq
    data analysis. Genome Biology, 17, 13.
Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based
    algorithm for discovering clusters in large spatial databases with
    noise. Proceedings of the Second International Conference on
    Knowledge Discovery and Data Mining, 226 to 231.
Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A
    review and recent developments. Philosophical Transactions of the
    Royal Society A, 374(2065), 1 to 16.
MacQueen, J. (1967). Some methods for classification and analysis of
    multivariate observations. Proceedings of the Fifth Berkeley
    Symposium on Mathematical Statistics and Probability, 1, 281 to 297.
Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the
    interpretation and validation of cluster analysis. Journal of
    Computational and Applied Mathematics, 20, 53 to 65.
Schmitz, R., Wright, G. W., Huang, D. W., Johnson, C. A., Phelan, J. D.,
    Wang, J. Q., Roulland, S., Kasbekar, M., Young, R. M., Shaffer, A. L.,
    Hodson, D. J., Xiao, W., Yu, X., Yang, Y., Zhao, H., Xu, W., Liu, X.,
    Zhou, B., Du, W., ... Staudt, L. M. (2018). Genetics and pathogenesis
    of diffuse large B-cell lymphoma. New England Journal of Medicine,
    378(15), 1396 to 1407.
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: PREPROCESSING - StandardScaler (from scratch)
# =============================================================================
# Standardizes features by removing the mean and scaling to unit variance,
# the same behavior as sklearn.preprocessing.StandardScaler. This matters
# for K-Means specifically because it is a distance-based algorithm: genes
# with a naturally larger numeric range would otherwise dominate the
# distance calculation regardless of their actual biological importance.

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
        std[std == 0] = 1.0  # guard against zero-variance (constant) columns
        self.scale_ = std
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler instance is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# =============================================================================
# SECTION 2: DIMENSIONALITY REDUCTION - PCA via numpy SVD (from scratch)
# =============================================================================
# Principal Component Analysis finds the orthogonal directions (components)
# along which the data varies the most and projects the data onto the top
# few. Used here ahead of K-Means to counter the curse of dimensionality:
# with thousands of gene features, Euclidean distance stops being a
# reliable way to tell observations apart (Jolliffe & Cadima, 2016).

class PCA:
    """Principal Component Analysis via SVD.

    Parameters
    ----------
    n_components : int or float, optional
        Number of components to keep. If a float in (0, 1], it is treated
        as the minimum proportion of cumulative explained variance to
        retain, and the number of components is chosen automatically.
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


# =============================================================================
# SECTION 3: CLUSTERING - K-Means, k-means++ init + Lloyd's algorithm (from scratch)
# =============================================================================
# The core algorithm this assignment requires: assign each point to its
# nearest centroid, recompute centroids as the mean of their assigned
# points, repeat until assignments stop changing (MacQueen, 1967).
# k-means++ initialization spreads the initial centroids out based on
# distance, which reduces the chance of converging to a poor local optimum
# compared to picking k random starting points.

def _pairwise_sq_dists(X, centers):
    """Squared Euclidean distance between every row of X and every center."""
    X_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]
    C_sq = np.sum(centers ** 2, axis=1)[np.newaxis, :]
    cross = X @ centers.T
    dists = X_sq - 2 * cross + C_sq
    return np.maximum(dists, 0.0)  # clip tiny negative values from floating-point error


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
        (within-cluster sum of squares) - the quantity K-Means minimizes.
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
            # Probability of being chosen as the next center is proportional
            # to squared distance from the closest existing center, this is
            # what spreads the initial centers out across the data.
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
            # Assignment step: each point joins its nearest centroid.
            dists = _pairwise_sq_dists(X, centers)
            new_labels = np.argmin(dists, axis=1)

            # Update step: recompute each centroid as the mean of its
            # assigned points.
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


# =============================================================================
# SECTION 4: EVALUATION METRICS (from scratch)
# =============================================================================
# silhouette_score is used to select k (Rousseeuw, 1987). cluster_purity and
# cross_tab are general-purpose label-agreement utilities kept here for
# completeness and used by the unit tests; the main DLBCL pipeline below
# uses marker-gene profiling instead, since this dataset has no
# ground-truth subtype label to compare clusters against.

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
    s(i) = (b(i) - a(i)) / max(a(i), b(i)). Returns the mean of s(i), a
    value from -1 (poor clustering) to +1 (dense, well-separated clusters).
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
    predicted cluster (a validation metric, never used during fitting)."""
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


def cross_tab(pred_labels, true_labels, pred_name="Cluster", true_name="Label"):
    """Contingency table of predicted cluster vs. a known label."""
    df = pd.DataFrame({pred_name: pred_labels, true_name: true_labels})
    return pd.crosstab(df[pred_name], df[true_name])


# =============================================================================
# SECTION 5: DATA LOADING - assembling the DLBCL gene expression matrix
# =============================================================================
# Each raw file downloaded by download_dlbcl_data.py is a GDC "augmented
# STAR gene counts" TSV: an optional leading comment line, a header row,
# four STAR alignment QC summary rows (N_unmapped, N_multimapping,
# N_noFeature, N_ambiguous, which are not genes and are dropped), then one
# row per gene. This section keeps only protein-coding genes, applies the
# standard log2(TPM + 1) variance-stabilizing transform (Conesa et al.,
# 2016), assembles every sample into one matrix, and keeps the top-N most
# variable genes (highly variable gene selection) to keep the feature
# space tractable ahead of PCA.

QC_PSEUDO_GENE_PREFIX = "N_"

# Genes with well-established roles in diffuse large B-cell lymphoma
# biology (germinal-center vs. activated B-cell programs, proliferation,
# and apoptosis regulation), used only to help interpret clusters after
# fitting, never during fitting itself (Schmitz et al., 2018).
MARKER_GENES = ["MYC", "BCL2", "BCL6", "MME", "IRF4", "CD19", "PAX5", "CCND2", "TP53"]


class DatasetNotFoundError(FileNotFoundError):
    pass


def _parse_gdc_gene_counts_file(path):
    """Parse one GDC augmented_star_gene_counts.tsv file.

    Returns a pandas.Series indexed by gene_id, containing log2(TPM + 1)
    values for protein-coding genes. Also attaches a parallel
    gene_id -> gene_name mapping as the Series' .attrs["gene_names"].
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # The header row is detected by content, not a fixed line number, so
    # this works whether or not the file has a leading "# gene-model: ..."
    # comment line.
    header_idx = None
    for i, line in enumerate(lines):
        if line.split("\t")[0].strip() == "gene_id":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find a 'gene_id' header row in {path}")

    header = [c.strip() for c in lines[header_idx].rstrip("\n").split("\t")]
    required = ["gene_id", "gene_name", "gene_type", "tpm_unstranded"]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"{path} is missing expected column(s) {missing}; found {header}")

    col_idx = {name: header.index(name) for name in required}

    gene_ids, gene_names, values = [], [], []
    for line in lines[header_idx + 1:]:
        line = line.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        gene_id = fields[col_idx["gene_id"]].strip()
        if gene_id.startswith(QC_PSEUDO_GENE_PREFIX):
            continue  # STAR QC summary row (N_unmapped, N_multimapping, ...), not a gene
        gene_type = fields[col_idx["gene_type"]].strip()
        if gene_type != "protein_coding":
            continue
        try:
            tpm = float(fields[col_idx["tpm_unstranded"]])
        except (ValueError, IndexError):
            continue
        gene_ids.append(gene_id)
        gene_names.append(fields[col_idx["gene_name"]].strip())
        values.append(np.log2(tpm + 1.0))

    series = pd.Series(values, index=pd.Index(gene_ids, name="gene_id"), name="log2_tpm")
    series.attrs["gene_names"] = dict(zip(gene_ids, gene_names))
    return series


def load_dlbcl_gene_expression(data_dir, top_n_genes=2000, min_samples=10):
    """Assemble the DLBCL samples-by-genes expression matrix.

    Parameters
    ----------
    data_dir : str
        Directory containing manifest.csv and a raw/ subfolder of
        <file_id>.tsv files, as produced by download_dlbcl_data.py.
    top_n_genes : int
        Number of most-variable protein-coding genes to keep.
    min_samples : int
        Minimum number of successfully parsed samples required.

    Returns
    -------
    features : pandas.DataFrame
        Shape (n_samples, top_n_genes), indexed by file_id.
    manifest : pandas.DataFrame
        Sample metadata (case_id, case_submitter_id, sample_type),
        indexed by file_id, aligned with features.
    gene_name_map : dict
        gene_id -> gene_name, for the genes retained in `features`.
    """
    manifest_path = os.path.join(data_dir, "manifest.csv")
    raw_dir = os.path.join(data_dir, "raw")

    if not os.path.exists(manifest_path) or not os.path.isdir(raw_dir):
        raise DatasetNotFoundError(
            f"Expected '{manifest_path}' and a 'raw/' folder of downloaded files under "
            f"'{data_dir}'. Run download_dlbcl_data.py first (see README.md)."
        )

    manifest = pd.read_csv(manifest_path).set_index("file_id")

    raw_files = glob.glob(os.path.join(raw_dir, "*.tsv"))
    if len(raw_files) < min_samples:
        raise DatasetNotFoundError(
            f"Only found {len(raw_files)} downloaded file(s) in '{raw_dir}'; need at least "
            f"{min_samples}. Run download_dlbcl_data.py (see README.md) to download the "
            "DLBCL gene expression files before running this pipeline."
        )

    per_sample_series = {}
    gene_name_map = {}
    for path in raw_files:
        file_id = os.path.splitext(os.path.basename(path))[0]
        try:
            s = _parse_gdc_gene_counts_file(path)
        except ValueError as e:
            print(f"  Skipping {file_id}: {e}")
            continue
        per_sample_series[file_id] = s
        gene_name_map.update(s.attrs["gene_names"])

    if len(per_sample_series) < min_samples:
        raise DatasetNotFoundError(
            f"Only {len(per_sample_series)} file(s) parsed successfully; need at least {min_samples}."
        )

    # Align on the gene panel shared by every sample.
    common_genes = None
    for s in per_sample_series.values():
        idx = set(s.index)
        common_genes = idx if common_genes is None else (common_genes & idx)
    common_genes = sorted(common_genes)

    matrix = pd.DataFrame({
        file_id: s.reindex(common_genes) for file_id, s in per_sample_series.items()
    }).T  # samples x genes
    matrix.index.name = "file_id"

    # Highly-variable gene selection: keep the top_n_genes with the
    # largest variance across samples.
    gene_variance = matrix.var(axis=0).sort_values(ascending=False)
    top_genes = gene_variance.head(min(top_n_genes, matrix.shape[1])).index.tolist()
    features = matrix[top_genes]

    manifest = manifest.reindex(features.index)
    trimmed_gene_name_map = {g: gene_name_map.get(g, g) for g in top_genes}

    return features, manifest, trimmed_gene_name_map


# =============================================================================
# SECTION 6: MAIN PIPELINE - ties every section above together
# =============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "data")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "output")

K_RANGE = range(2, 9)
PCA_VARIANCE_TARGET = 0.90
RANDOM_STATE = 42
TOP_N_GENES = 2000


def run_pipeline(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR, k_override=None,
                  top_n_genes=TOP_N_GENES):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/7] Loading and assembling DLBCL gene expression matrix from: {data_dir}")
    features, manifest, gene_name_map = load_dlbcl_gene_expression(data_dir, top_n_genes=top_n_genes)
    print(f"      Assembled {features.shape[0]} samples x {features.shape[1]} genes "
          f"(top {top_n_genes} most variable protein-coding genes, log2(TPM+1)).")

    print("[2/7] Standardizing features (zero mean, unit variance)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    print(f"[3/7] Reducing dimensionality with PCA (target {PCA_VARIANCE_TARGET:.0%} variance)...")
    pca = PCA(n_components=PCA_VARIANCE_TARGET)
    X_reduced = pca.fit_transform(X_scaled)
    n_components = X_reduced.shape[1]
    cum_var = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    print(f"      Reduced to {n_components} components, explaining {cum_var:.1%} of variance.")

    print(f"[4/7] Running elbow method and silhouette analysis over k = {list(K_RANGE)}...")
    inertias, silhouettes = [], []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=8, random_state=RANDOM_STATE)
        model.fit(X_reduced)
        inertias.append(model.inertia_)
        sil = silhouette_score(X_reduced, model.labels_, sample_size=min(600, X_reduced.shape[0]),
                                random_state=RANDOM_STATE)
        silhouettes.append(sil)
        print(f"      k={k}: inertia={model.inertia_:,.1f}, silhouette={sil:.3f}")

    best_k = k_override if k_override is not None else list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"[5/7] Selected k = {best_k} (highest silhouette score).")

    final_model = KMeans(n_clusters=best_k, n_init=15, random_state=RANDOM_STATE)
    final_model.fit(X_reduced)
    final_silhouette = silhouette_score(X_reduced, final_model.labels_,
                                         sample_size=min(600, X_reduced.shape[0]),
                                         random_state=RANDOM_STATE)

    print("[6/7] Characterizing clusters by DLBCL marker gene expression (post-hoc interpretation only)...")
    marker_profile = _marker_gene_profile(features, final_model.labels_, gene_name_map)
    print(marker_profile.round(2))

    print("[7/7] Saving plots and results to output/ ...")
    _plot_elbow(list(K_RANGE), inertias, output_dir)
    _plot_silhouette(list(K_RANGE), silhouettes, output_dir)
    _plot_pca_scatter(X_reduced, final_model.labels_, output_dir)

    results = {
        "n_samples": int(features.shape[0]),
        "n_genes_selected": int(features.shape[1]),
        "n_pca_components": int(n_components),
        "pca_cumulative_variance_explained": round(cum_var, 4),
        "k_range_tested": list(K_RANGE),
        "inertia_by_k": {str(k): round(v, 2) for k, v in zip(K_RANGE, inertias)},
        "silhouette_by_k": {str(k): round(v, 4) for k, v in zip(K_RANGE, silhouettes)},
        "selected_k": int(best_k),
        "final_inertia": round(final_model.inertia_, 2),
        "final_silhouette_score": round(final_silhouette, 4),
        "marker_gene_profile_by_cluster": marker_profile.round(3).to_dict(),
        "cluster_sizes": {str(c): int(n) for c, n in
                           zip(*np.unique(final_model.labels_, return_counts=True))},
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    assignments = manifest.copy()
    assignments["cluster"] = final_model.labels_
    assignments.to_csv(os.path.join(output_dir, "cluster_assignments.csv"))
    marker_profile.to_csv(os.path.join(output_dir, "marker_gene_profile_by_cluster.csv"))

    print("Done. Results written to:", output_dir)
    return results


def _marker_gene_profile(features, labels, gene_name_map):
    """Mean standardized expression of each available marker gene, per
    cluster. Standardized (z-scored) so genes with very different
    expression scales are comparable within the table."""
    name_to_id = {}
    for gene_id, gene_name in gene_name_map.items():
        name_to_id.setdefault(gene_name, gene_id)

    available = [g for g in MARKER_GENES if g in name_to_id]
    if not available:
        return pd.DataFrame()

    marker_ids = [name_to_id[g] for g in available]
    marker_expr = features[marker_ids].copy()
    marker_expr.columns = available
    z = (marker_expr - marker_expr.mean()) / marker_expr.std(ddof=0).replace(0, 1)
    z["cluster"] = labels
    return z.groupby("cluster").mean()


def _plot_elbow(ks, inertias, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(ks, inertias, marker="o", color="#2E5C8A")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow Method for Selecting k")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "elbow_plot.png"))
    plt.close(fig)


def _plot_silhouette(ks, silhouettes, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(ks, silhouettes, marker="o", color="#8A6D00")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Silhouette Score by k")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "silhouette_plot.png"))
    plt.close(fig)


def _plot_pca_scatter(X_reduced, pred_labels, output_dir):
    if X_reduced.shape[1] > 2:
        viz_pca = PCA(n_components=2)
        X_2d = viz_pca.fit_transform(X_reduced)
    else:
        X_2d = X_reduced

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=pred_labels, cmap="tab10", s=16, alpha=0.85)
    ax.set_title("K-Means Clusters of DLBCL Tumor Samples (PCA Projection)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_cluster_scatter.png"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLBCL gene expression K-Means clustering application")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=None, help="Override automatic k selection")
    parser.add_argument("--top-n-genes", type=int, default=TOP_N_GENES)
    args = parser.parse_args()

    try:
        run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, k_override=args.k,
                     top_n_genes=args.top_n_genes)
    except DatasetNotFoundError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
