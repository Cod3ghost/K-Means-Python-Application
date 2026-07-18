"""
test_kmeans_app.py

Unit tests for the K-Means Python Application. Uses synthetic data
generated with numpy (well-separated Gaussian blobs) plus hand-built
GDC-format fixture files, so the test suite is self-contained and does
not depend on the real DLBCL dataset being downloaded.

Run with:
    python3 -m unittest test_kmeans_app.py -v

Author: Desmond Amos Bature
Course: BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment - K-Means Python Application
"""

import csv
import os
import tempfile
import unittest

import numpy as np

from preprocessing import StandardScaler
from pca import PCA
from kmeans import KMeans
from metrics import silhouette_score, cluster_purity, cross_tab
from data_loader import load_dlbcl_gene_expression, _parse_gdc_gene_counts_file, DatasetNotFoundError


def make_blobs(n_per_cluster=60, centers=((0, 0), (10, 10), (10, -10)), std=0.8, random_state=0):
    """Minimal stand-in for sklearn.datasets.make_blobs (unavailable here)."""
    rng = np.random.default_rng(random_state)
    X_parts, y_parts = [], []
    for i, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=std, size=(n_per_cluster, len(c)))
        X_parts.append(pts)
        y_parts.append(np.full(n_per_cluster, i))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


class TestStandardScaler(unittest.TestCase):
    def test_output_has_zero_mean_unit_variance(self):
        rng = np.random.default_rng(1)
        X = rng.normal(loc=[5, -3, 100], scale=[2, 0.5, 50], size=(200, 3))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-8)

    def test_constant_column_does_not_raise_or_produce_nan(self):
        X = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.assertFalse(np.isnan(X_scaled).any())
        np.testing.assert_allclose(X_scaled[:, 0], 0.0)

    def test_transform_before_fit_raises(self):
        scaler = StandardScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(np.array([[1.0, 2.0]]))


class TestPCA(unittest.TestCase):
    def test_reduces_dimensionality(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 10))
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X)
        self.assertEqual(X_reduced.shape, (100, 3))

    def test_explained_variance_ratio_sums_leq_one(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 8))
        pca = PCA(n_components=None)
        pca.fit(X)
        self.assertLessEqual(pca.explained_variance_ratio_.sum(), 1.0 + 1e-8)

    def test_first_component_captures_dominant_direction(self):
        rng = np.random.default_rng(4)
        X = np.column_stack([
            rng.normal(scale=10.0, size=500),
            rng.normal(scale=0.5, size=500),
        ])
        pca = PCA(n_components=2)
        pca.fit(X)
        self.assertGreater(pca.explained_variance_ratio_[0], 0.9)

    def test_variance_target_selects_fewer_components_than_all(self):
        rng = np.random.default_rng(5)
        base = rng.normal(size=(200, 1))
        noise = rng.normal(scale=0.01, size=(200, 9))
        X = np.hstack([base, base, base, noise])
        pca = PCA(n_components=0.90)
        X_reduced = pca.fit_transform(X)
        self.assertLess(X_reduced.shape[1], X.shape[1])


class TestKMeans(unittest.TestCase):
    def test_recovers_known_cluster_count_on_separated_blobs(self):
        X, y_true = make_blobs(n_per_cluster=50, centers=((0, 0), (12, 12), (12, -12)),
                                std=0.6, random_state=10)
        model = KMeans(n_clusters=3, n_init=10, random_state=10)
        model.fit(X)
        self.assertEqual(len(np.unique(model.labels_)), 3)
        purity = cluster_purity(model.labels_, y_true)
        self.assertGreater(purity, 0.95)

    def test_inertia_decreases_as_k_increases(self):
        X, _ = make_blobs(n_per_cluster=40, random_state=11)
        inertias = []
        for k in (1, 2, 3, 4):
            model = KMeans(n_clusters=k, n_init=5, random_state=11)
            model.fit(X)
            inertias.append(model.inertia_)
        for i in range(len(inertias) - 1):
            self.assertLessEqual(inertias[i + 1], inertias[i] + 1e-6)

    def test_predict_matches_fit_labels_on_training_data(self):
        X, _ = make_blobs(n_per_cluster=30, random_state=12)
        model = KMeans(n_clusters=3, n_init=5, random_state=12)
        model.fit(X)
        predicted = model.predict(X)
        np.testing.assert_array_equal(predicted, model.labels_)

    def test_raises_when_k_exceeds_n_samples(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        model = KMeans(n_clusters=5)
        with self.assertRaises(ValueError):
            model.fit(X)

    def test_no_cluster_is_left_empty(self):
        X, _ = make_blobs(n_per_cluster=15, centers=((0, 0), (20, 20)), std=0.3, random_state=13)
        model = KMeans(n_clusters=4, n_init=10, random_state=13)
        model.fit(X)
        counts = np.bincount(model.labels_, minlength=4)
        self.assertTrue(np.all(counts > 0))


class TestMetrics(unittest.TestCase):
    def test_silhouette_score_high_for_well_separated_clusters(self):
        X, y_true = make_blobs(n_per_cluster=40, centers=((0, 0), (15, 15)), std=0.5, random_state=20)
        score = silhouette_score(X, y_true)
        self.assertGreater(score, 0.8)

    def test_silhouette_score_low_for_overlapping_random_labels(self):
        rng = np.random.default_rng(21)
        X = rng.normal(size=(100, 2))
        random_labels = rng.integers(0, 3, size=100)
        score = silhouette_score(X, random_labels)
        self.assertLess(score, 0.3)

    def test_purity_is_perfect_when_clusters_match_labels_exactly(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(cluster_purity(labels, labels), 1.0)

    def test_cross_tab_shape(self):
        pred = np.array([0, 0, 1, 1, 2, 2])
        true = np.array(["A", "A", "B", "B", "C", "C"])
        table = cross_tab(pred, true)
        self.assertEqual(table.shape, (3, 3))


def _write_fixture_gdc_file(path, sample_seed, gene_defs, include_comment_line=True):
    rng = np.random.default_rng(sample_seed)
    lines = []
    if include_comment_line:
        lines.append("# gene-model: GENCODE v36")
    lines.append("gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\t"
                  "tpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded")
    for qc_name in ("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"):
        lines.append(f"{qc_name}\t\t\t{rng.integers(100, 1000)}\t0\t0\t\t\t")
    for gene_id, gene_name, gene_type, base_tpm in gene_defs:
        tpm = max(0.0, base_tpm + rng.normal(scale=base_tpm * 0.3 + 0.1))
        lines.append(f"{gene_id}\t{gene_name}\t{gene_type}\t{int(tpm*10)}\t0\t0\t{tpm:.4f}\t{tpm*0.9:.4f}\t{tpm*0.95:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


GENE_DEFS = [
    ("ENSG00000001", "MYC", "protein_coding", 15.0),
    ("ENSG00000002", "BCL2", "protein_coding", 8.0),
    ("ENSG00000003", "BCL6", "protein_coding", 12.0),
    ("ENSG00000004", "MME", "protein_coding", 5.0),
    ("ENSG00000005", "IRF4", "protein_coding", 20.0),
    ("ENSG00000006", "CD19", "protein_coding", 30.0),
    ("ENSG00000007", "PAX5", "protein_coding", 25.0),
    ("ENSG00000008", "CCND2", "protein_coding", 18.0),
    ("ENSG00000009", "TP53", "protein_coding", 6.0),
    ("ENSG00000010", "LINC00001", "lncRNA", 3.0),
    ("ENSG00000011", "MIR0001", "miRNA", 1.0),
    ("ENSG00000012", "GENEX", "protein_coding", 9.0),
]


class TestDataLoader(unittest.TestCase):
    def test_parse_single_file_keeps_only_protein_coding_genes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample0.tsv")
            _write_fixture_gdc_file(path, sample_seed=0, gene_defs=GENE_DEFS)
            series = _parse_gdc_gene_counts_file(path)
            self.assertNotIn("ENSG00000010", series.index)
            self.assertNotIn("ENSG00000011", series.index)
            self.assertIn("ENSG00000001", series.index)
            self.assertEqual(len(series), 10)

    def test_parse_handles_missing_comment_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample0.tsv")
            _write_fixture_gdc_file(path, sample_seed=1, gene_defs=GENE_DEFS, include_comment_line=False)
            series = _parse_gdc_gene_counts_file(path)
            self.assertEqual(len(series), 10)

    def test_qc_pseudo_rows_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample0.tsv")
            _write_fixture_gdc_file(path, sample_seed=2, gene_defs=GENE_DEFS)
            series = _parse_gdc_gene_counts_file(path)
            for qc_name in ("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"):
                self.assertNotIn(qc_name, series.index)

    def test_load_dlbcl_gene_expression_assembles_matrix_across_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = os.path.join(tmp, "raw")
            os.makedirs(raw_dir)
            file_ids = [f"file-{i}" for i in range(12)]
            for i, file_id in enumerate(file_ids):
                _write_fixture_gdc_file(os.path.join(raw_dir, f"{file_id}.tsv"), sample_seed=i,
                                         gene_defs=GENE_DEFS)

            manifest_path = os.path.join(tmp, "manifest.csv")
            with open(manifest_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["file_id", "file_name", "file_size",
                                                        "case_id", "case_submitter_id", "sample_type"])
                writer.writeheader()
                for file_id in file_ids:
                    writer.writerow({"file_id": file_id, "file_name": f"{file_id}.tsv", "file_size": 100,
                                      "case_id": f"case-{file_id}", "case_submitter_id": f"case-{file_id}",
                                      "sample_type": "Primary Tumor"})

            features, manifest, gene_name_map = load_dlbcl_gene_expression(
                tmp, top_n_genes=5, min_samples=10)

            self.assertEqual(features.shape[0], 12)
            self.assertEqual(features.shape[1], 5)
            self.assertEqual(len(manifest), 12)
            self.assertTrue(set(gene_name_map.keys()).issubset(set(GENE_DEFS[i][0] for i in range(10))))

    def test_raises_when_too_few_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = os.path.join(tmp, "raw")
            os.makedirs(raw_dir)
            _write_fixture_gdc_file(os.path.join(raw_dir, "only-one.tsv"), sample_seed=0, gene_defs=GENE_DEFS)
            manifest_path = os.path.join(tmp, "manifest.csv")
            with open(manifest_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["file_id", "file_name", "file_size",
                                                        "case_id", "case_submitter_id", "sample_type"])
                writer.writeheader()
                writer.writerow({"file_id": "only-one", "file_name": "only-one.tsv", "file_size": 100,
                                  "case_id": "case-1", "case_submitter_id": "case-1",
                                  "sample_type": "Primary Tumor"})

            with self.assertRaises(DatasetNotFoundError):
                load_dlbcl_gene_expression(tmp, top_n_genes=5, min_samples=10)

    def test_raises_when_data_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_dir = os.path.join(tmp, "does_not_exist")
            with self.assertRaises(DatasetNotFoundError):
                load_dlbcl_gene_expression(empty_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
