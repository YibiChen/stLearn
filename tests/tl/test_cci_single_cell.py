"""Tests for single-cell resolution CCI (st.tl.cci.run_sc / run_cci_sc)."""

import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import cKDTree

import stlearn as st
from stlearn.tl.cci.single_cell import radius_graph
from stlearn.tl.cci.spatial_null import (
    label_morans_i,
    permute_labels,
    permute_values,
    values_morans_i,
)


def make_tissue(n: int = 3000, seed: int = 0) -> tuple[AnnData, np.ndarray]:
    """Cells in patches of types A-D; LA is expressed by A cells, RB by B
    cells, so A -> B is the only real interaction."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 800, (n, 2))
    centres = rng.uniform(0, 800, (40, 2))
    types = np.array(list("ABCD"))[rng.integers(0, 4, 40)]
    labels = types[cKDTree(centres).query(xy)[1]]
    n_bg = 150
    bg = rng.poisson(rng.gamma(0.6, 1.0, n_bg)[None, :], (n, n_bg)).astype(float)
    la = np.where(labels == "A", rng.poisson(2.0, n), 0).astype(float)
    rb = np.where(labels == "B", rng.poisson(2.0, n), 0).astype(float)
    expr = np.log1p(np.hstack([bg, la[:, None], rb[:, None]]))
    adata = AnnData(
        sp.csr_matrix(expr),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(labels)},
            index=[f"cell{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_bg)] + ["LA", "RB"]),
    )
    adata.obsm["spatial"] = xy
    return adata, np.array(["LA_RB"])


class TestRadiusGraph(unittest.TestCase):
    def test_exact_radius_no_self(self):
        coords = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 5.01], [0.0, 0.0]])
        graph = radius_graph(coords, 5.0).toarray()
        self.assertTrue(np.all(np.diag(graph) == 0))
        self.assertTrue(np.array_equal(graph, graph.T))
        self.assertEqual(graph[0, 1], 1)  # distance exactly 5
        self.assertEqual(graph[0, 2], 0)  # distance 5.01
        self.assertEqual(graph[0, 3], 1)  # same position, different cell


class TestScoring(unittest.TestCase):
    def test_sender_and_receiver_are_different_cells(self):
        # Cell 0 expresses both L and R but has no neighbours: no autocrine
        # score. Cell 1 sends L to cell 2, which expresses R.
        coords = np.array([[0.0, 0.0], [100.0, 0.0], [105.0, 0.0]])
        n_bg = 60
        rng = np.random.default_rng(1)
        bg = rng.poisson(1.0, (3, n_bg)).astype(float)
        lr = np.array([[2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        adata = AnnData(
            np.hstack([bg, lr]),
            var=pd.DataFrame(index=[f"G{i}" for i in range(n_bg)] + ["L1", "R1"]),
        )
        adata.obsm["spatial"] = coords
        st.tl.cci.run_sc(
            adata,
            np.array(["L1_R1"]),
            radius=10,
            min_cells=1,
            n_pairs=200,
            verbose=False,
        )
        send = adata.obsm["lr_scores"].toarray()[:, 0]
        recv = adata.obsm["lr_receiver_scores"].toarray()[:, 0]
        self.assertEqual(send[0], 0)
        self.assertGreater(send[1], 0)
        self.assertEqual(send[2], 0)
        self.assertGreater(recv[2], 0)

    def test_different_cell_types_drops_same_type_neighbours(self):
        adata, lrs = make_tissue(n=800)
        st.tl.cci.run_sc(
            adata,
            lrs,
            radius=40,
            use_label="cell_type",
            different_cell_types=True,
            n_pairs=200,
            verbose=False,
        )
        coo = adata.obsp["cci_radius_graph"].tocoo()
        codes = adata.obs["cell_type"].cat.codes.values
        self.assertTrue(np.all(codes[coo.row] != codes[coo.col]))


class TestSpatialNull(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        n = 2000
        self.coords = rng.uniform(0, 600, (n, 2))
        self.graph = radius_graph(self.coords, 25)
        centres = rng.uniform(0, 600, (25, 2))
        self.codes = rng.integers(0, 3, 25)[
            cKDTree(centres).query(self.coords)[1]
        ].astype(np.int32)

    def test_labels_keep_composition_and_autocorrelation(self):
        perms, achieved, observed = permute_labels(
            self.codes, self.graph, 4, method="spatial"
        )
        for perm, ach in zip(perms, achieved):
            self.assertTrue(np.array_equal(np.bincount(perm), np.bincount(self.codes)))
            # The tracked Moran's I matches a fresh computation.
            np.testing.assert_allclose(
                label_morans_i(perm, self.graph, 3), ach, atol=1e-8
            )
        self.assertLess(np.abs(achieved - observed).max(), 0.011)

        _, achieved_random, _ = permute_labels(
            self.codes, self.graph, 4, method="random"
        )
        self.assertLess(np.abs(achieved_random).max(), 0.1)

    def test_values_keep_distribution_and_autocorrelation(self):
        dom = np.linalg.norm(self.coords - 300, axis=1) < 150
        values = np.where(dom, np.log1p(np.arange(len(dom)) % 4), 0.0)
        surrogates, achieved, observed = permute_values(
            values, self.graph, 3, coords=self.coords, radius=25
        )
        for surrogate, ach in zip(surrogates, achieved):
            np.testing.assert_array_equal(np.sort(surrogate), np.sort(values))
            np.testing.assert_allclose(
                values_morans_i(surrogate, self.graph, self.coords, 25),
                ach,
                atol=1e-8,
            )
        self.assertLess(np.abs(achieved - observed).max(), 0.02)


class TestRunCCISingleCell(unittest.TestCase):
    def test_detects_sender_to_receiver_type(self):
        adata, lrs = make_tissue()
        st.tl.cci.run_sc(adata, lrs, radius=30, n_pairs=200, verbose=False)
        summary = adata.uns["lr_summary"]
        self.assertGreater(summary.loc["LA_RB", "n_spots_sig"], 0)

        st.tl.cci.run_cci_sc(adata, "cell_type", n_perms=50, verbose=False)
        pvals = adata.uns["per_lr_cci_pvals_cell_type"]["LA_RB"]
        raw = adata.uns["per_lr_cci_raw_cell_type"]["LA_RB"]
        self.assertLess(pvals.loc["A", "B"], 0.05)
        # Directed: senders are A cells (ligand), receivers B cells (receptor).
        self.assertGreater(raw.loc["A", "B"], 0)
        self.assertEqual(raw.loc["B", "A"], 0)
        null = adata.uns["cci_sc_null_cell_type"]
        self.assertLess(null["null_max_abs_gap"].max(), 0.011)
