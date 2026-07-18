# DLBCL Gene Expression K-Means Python Application

BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment: K-Means Python Application
Desmond Amos Bature

## What this is

An unsupervised K-Means clustering application that groups diffuse large
B-cell lymphoma (DLBCL) tumor samples by their gene expression profiles.
K-Means and PCA are implemented from scratch in numpy (`src/kmeans.py`,
`src/pca.py`) rather than imported from scikit-learn, because scikit-learn
could not be installed in the sandbox used to build this project (no
package-index access, no root). The from-scratch implementation follows
the same algorithms scikit-learn uses (k-means++ initialization, Lloyd's
algorithm, SVD-based PCA) and is unit tested in `tests/`.

## Dataset

**National Cancer Institute Center for Cancer Research - Diffuse Large B
Cell Lymphoma (DLBCL) Genomics and Expression** (project NCICCR-DLBCL),
registered on the AWS Registry of Open Data
(`arn:aws:s3:::gdc-nciccr-phs001444-2-open`) and served by the Genomic
Data Commons (GDC). This project profiles 574 DLBCL tumors by exome and
transcriptome sequencing (Schmitz et al., 2018). The open-access tier
includes one RNA-Seq gene expression quantification file per sample
(`*.rna_seq.augmented_star_gene_counts.tsv`); everything else in the
project (aligned reads, fusion calls) is controlled-access and is not
used here.

### How to get the data

GDC's own HTTPS API serves these open files with no AWS account, no AWS
CLI, and no authentication required, so `download_dlbcl_data.py` (plain
Python standard library, no extra installs) does the whole job:

```bash
python download_dlbcl_data.py
```

This queries the GDC API for every open Gene Expression Quantification
file in project NCICCR-DLBCL (roughly 500 files, ~2 GB total), writes
`data/manifest.csv`, and downloads each file to `data/raw/<file_id>.tsv`.
It skips files it has already downloaded, so it is safe to re-run if
interrupted.

For a quick test run before committing to the full download:

```bash
python download_dlbcl_data.py --limit 30
```

## Project structure

```
TCGA_KMeans_App/
  download_dlbcl_data.py   <- run this first: downloads the dataset from GDC
  data/
    manifest.csv            <- created by download_dlbcl_data.py
    raw/                    <- created by download_dlbcl_data.py: one .tsv per sample
  src/
    preprocessing.py        <- StandardScaler (from scratch)
    pca.py                   <- PCA via numpy SVD (from scratch)
    kmeans.py                <- K-Means, k-means++ init, Lloyd's algorithm (from scratch)
    metrics.py                <- silhouette score, purity, cross-tab (from scratch)
    data_loader.py             <- parses and assembles the per-sample GDC files into one matrix
    main.py                     <- end-to-end pipeline, entry point
  tests/
    test_kmeans_app.py        <- 22 unit tests (synthetic fixtures, no download required)
    unit_test_results.log     <- captured passing test run
  output/                      <- created by main.py: plots, results.json, cluster_assignments.csv
```

## Running in VS Code

1. Open the `TCGA_KMeans_App` folder in VS Code.
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
3. Download the dataset (see above):
   ```bash
   python download_dlbcl_data.py
   ```
4. Run the pipeline:
   ```bash
   cd src
   python main.py
   ```
5. Run the unit tests (from the `TCGA_KMeans_App` root, with the venv active):
   ```bash
   cd src
   python -m unittest discover -s ../tests -p "test_*.py" -v
   ```
   Or use VS Code's built-in Test Explorer: Command Palette -> "Python:
   Configure Tests" -> unittest -> tests directory -> pattern `test_*.py`.

This also works the same way in PyCharm: mark `src` as a Sources Root,
right-click `main.py` to run it, right-click `tests/test_kmeans_app.py`
to run the test suite.

## Method summary

1. **Load & assemble** — `data_loader.py` reads the per-sample GDC
   `*.rna_seq.augmented_star_gene_counts.tsv` files listed in
   `data/manifest.csv` and assembles them into one samples x genes matrix
   (log2(TPM+1)), keeping the top 2,000 most variable protein-coding
   genes.
2. **Standardize** — `preprocessing.StandardScaler` rescales every gene
   to zero mean and unit variance so no single gene dominates on scale
   alone.
3. **Reduce dimensionality** — `pca.PCA` (SVD-based, from scratch)
   projects the standardized matrix down to the number of principal
   components needed to explain 90% of the variance.
4. **Choose k** — `kmeans.KMeans` is fit for every k in 2-8; the elbow
   method (inertia vs. k) and the mean silhouette score
   (`metrics.silhouette_score`) are plotted and compared to pick a
   defensible number of clusters.
5. **Fit the final model** — `KMeans` is refit at the selected k with
   multiple k-means++ initializations, keeping the best (lowest-inertia)
   run.
6. **Characterize clusters** — after fitting, each cluster's mean
   standardized expression of a curated panel of DLBCL marker genes
   (MYC, BCL2, BCL6, MME, IRF4, CD19, PAX5, CCND2) is computed as a
   post-hoc, domain-informed sanity check. No subtype or diagnosis label
   is used anywhere during fitting.
7. **Save outputs** — plots and a results summary are written to
   `output/`.

## Results (481-sample open-access cohort, top 2,000 genes)

Running the full pipeline against all 481 downloaded samples produced:

| Metric | Value |
| --- | --- |
| Samples x genes (input matrix) | 481 x 2,000 |
| PCA components (90% variance target) | 172 (explains 90.05% of variance) |
| k values tested | 2-8 |
| Selected k (highest silhouette score) | 2 |
| Final silhouette score | 0.098 |
| Final inertia | 771,490.09 |
| Cluster sizes | Cluster 0: 262 samples, Cluster 1: 219 samples |

Post-hoc marker gene profiling shows the two clusters separate along a
germinal-center-like vs. non-germinal-center-like axis consistent with
known DLBCL cell-of-origin biology: cluster 0 shows relatively higher
MYC, BCL2, IRF4, and CD19 expression, while cluster 1 shows relatively
higher BCL6, MME, and CCND2 expression. Full per-k inertia/silhouette
values and the per-cluster marker profile are in `output/results.json`;
per-sample cluster assignments are in
`output/cluster_assignments.csv`.

Silhouette scores in the 0.06-0.10 range are modest, which is expected:
real transcriptomic cohorts rarely form the tight, well-separated
clusters that synthetic benchmark data does, and DLBCL subtypes are
known from the literature to overlap on the continuum between
germinal-center B-cell and activated B-cell profiles rather than form
sharply distinct groups.

## References

- Alizadeh, A. A., et al. (2000). Distinct types of diffuse large B-cell
  lymphoma identified by gene expression profiling. *Nature*, 403,
  503-511.
- Schmitz, R., et al. (2018). Genetics and pathogenesis of diffuse large
  B-cell lymphoma. *New England Journal of Medicine*, 378(15),
  1396-1407.
- National Cancer Institute Center for Cancer Research - DLBCL Genomics
  and Expression (NCICCR-DLBCL), Genomic Data Commons,
  https://portal.gdc.cancer.gov/projects/NCICCR-DLBCL