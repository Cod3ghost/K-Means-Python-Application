"""
data_loader.py

Loads and assembles the NCICCR-DLBCL open RNA-Seq gene expression dataset
from the individual per-sample files downloaded by download_dlbcl_data.py.

Each raw file is a GDC "augmented STAR gene counts" TSV: an optional
leading comment line, a header row, four STAR alignment QC summary rows
(gene_id values N_unmapped, N_multimapping, N_noFeature, N_ambiguous,
which are not genes and are dropped), followed by one row per gene with
gene_id, gene_name, gene_type, raw counts, and TPM/FPKM columns.

This loader:
  1. Parses every raw/<file_id>.tsv file.
  2. Keeps only protein-coding genes (gene_type == "protein_coding"),
     since non-coding and pseudogene entries add noise without adding
     much biologically interpretable signal for a clustering exercise.
  3. Uses the tpm_unstranded column and applies a log2(TPM + 1) transform,
     the standard variance-stabilizing transform for expression data
     before distance-based methods like K-Means (Conesa et al., 2016).
  4. Assembles all samples into one samples-by-genes matrix, aligned on
     the shared gene panel across files.
  5. Reduces to the top-N most variable genes across samples (highly
     variable gene selection), which is both a standard genomics
     preprocessing step and necessary to keep the feature space
     tractable ahead of PCA.

There is no ground-truth subtype label shipped with this project (unlike
the earlier TCGA-PANCAN dataset, where cancer type was a built-in column),
so this loader returns a manifest of sample metadata (case ID, sample
type) rather than a label column. Cluster interpretation instead relies
on the expression of curated DLBCL marker genes (see MARKER_GENES below).

Author: Desmond Amos Bature
Course: BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment - K-Means Python Application
"""

import os
import glob

import numpy as np
import pandas as pd

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
