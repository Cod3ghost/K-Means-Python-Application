"""
main.py

K-Means Python Application - end-to-end pipeline.

Business framing
-----------------
Diffuse large B-cell lymphoma (DLBCL) is the most common aggressive
non-Hodgkin lymphoma, and it is well established in the literature that
DLBCL is not one disease but a biologically heterogeneous group of tumors
that respond differently to the same standard chemotherapy regimen
(Schmitz et al., 2018). This application clusters 574 DLBCL tumor samples
from the NCI Center for Cancer Research's open RNA-Seq gene expression
dataset (hosted on the AWS Registry of Open Data and the Genomic Data
Commons) using unsupervised K-Means, without using any diagnosis or
subtype label during fitting, to ask a realistic molecular-pathology
analytics question: does gene expression alone separate this patient
cohort into distinct groups, the same starting point real DLBCL subtype
classifiers (cell-of-origin, genetic subtype) are built from? Because the
public open-access tier of this dataset does not ship a subtype label,
clusters are interpreted after the fact using the expression of a small
panel of genes with known roles in B-cell lymphoma biology (MYC, BCL2,
BCL6, and related markers), the same way a business or clinical analyst
would sanity-check an unsupervised segmentation against known domain
signals when no ground-truth label is available.

Pipeline
--------
1. Load the assembled DLBCL gene expression matrix (samples x top
   variable protein-coding genes), produced by data_loader.py from the
   raw per-sample files downloaded by download_dlbcl_data.py.
2. Standardize features (zero mean, unit variance).
3. Reduce dimensionality with PCA, both to counter the curse of
   dimensionality at thousands of gene features and because Module 4
   pairs clustering with dimensionality reduction.
4. Use the elbow method (inertia vs. k) and the silhouette score to
   choose a defensible number of clusters, k.
5. Fit the final from-scratch K-Means model at the chosen k.
6. Characterize each cluster by its mean expression of curated DLBCL
   marker genes, as a post-hoc, domain-informed interpretation step
   (no ground-truth subtype label is used at any point during fitting).
7. Save plots and a results summary to the output/ folder.

Author: Desmond Amos Bature
Course: BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment - K-Means Python Application
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_dlbcl_gene_expression, DatasetNotFoundError, MARKER_GENES
from preprocessing import StandardScaler
from pca import PCA
from kmeans import KMeans
from metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "..", "data")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "..", "output")

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
