"""Tests for the grid communication potential (st.tl.cci.run_cci_potential)."""

import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

import stlearn as st


def make_tissue(n: int = 300, seed: int = 0) -> AnnData:
    """Random cells of types A-C on a 200 x 200 um square; Poisson counts of L, R
    and a background gene."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson([1.0, 0.8, 5.0], (n, 3)).astype(float)
    counts[:, 2] += 1  # every cell has a positive library size
    adata = AnnData(
        sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(rng.choice(list("ABC"), n))},
            index=[f"cell{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["L", "R", "BG"]),
    )
    adata.obsm["spatial"] = rng.uniform(0, 200, (n, 2))
    return adata


def brute_force(adata, sender, receiver, bin_size, n_origins):
    """Pool every ordered pair of distinct cells that share a bin, grid by grid."""
    counts = adata.X.toarray()
    library = counts.sum(axis=1)
    labels = adata.obs["cell_type"].to_numpy()
    coords = adata.obsm["spatial"]
    per_axis = int(np.sqrt(n_origins))
    numerator = opportunity = 0.0
    for dy in range(per_axis):
        for dx in range(per_axis):
            shift = bin_size * np.array([dx, dy]) / per_axis
            bin_keys = np.floor((coords - coords.min(axis=0) + shift) / bin_size)
            bin_keys = [tuple(key) for key in bin_keys.astype(int)]
            for key in set(bin_keys):
                in_bin = [i for i, k in enumerate(bin_keys) if k == key]
                pairs = [
                    (i, j)
                    for i in in_bin
                    for j in in_bin
                    if i != j and labels[i] == sender and labels[j] == receiver
                ]
                if not pairs:
                    continue
                pair_counts = sum(counts[i, 0] * counts[j, 1] for i, j in pairs)
                pair_exposure = sum(library[i] * library[j] for i, j in pairs)
                numerator += len(pairs) * pair_counts / pair_exposure
                opportunity += len(pairs)
    return np.median(library) ** 2 * numerator / opportunity


def potential_table(adata, **kwargs):
    st.tl.cci.run_cci_potential(
        adata, ["L_R"], use_label="cell_type", verbose=False, **kwargs
    )
    return adata.uns["cci_potential_cell_type"].set_index(["sender", "receiver"])


class TestCCIPotential(unittest.TestCase):
    def test_matches_brute_force(self):
        adata = make_tissue()
        for n_origins in (1, 4):
            table = potential_table(adata, bin_size=50, n_origins=n_origins)
            for sender, receiver in [("A", "B"), ("B", "A"), ("A", "A")]:
                expected = brute_force(adata, sender, receiver, 50, n_origins)
                observed = table.loc[(sender, receiver), "potential"]
                self.assertAlmostEqual(observed, expected, places=10)

    def test_no_shared_bin_is_nan_and_no_expression_is_zero(self):
        adata = make_tissue()
        labels = np.where(adata.obsm["spatial"][:, 0] < 100, "A", "B")
        adata.obs["cell_type"] = pd.Categorical(labels)
        counts = adata.X.toarray()
        counts[labels == "A", 0] = 0  # A expresses no ligand
        adata.X = sp.csr_matrix(counts)
        # A and B share the bins along x = 100
        table = potential_table(adata, bin_size=20, n_origins=1)
        self.assertEqual(table.loc[("A", "B"), "potential"], 0)

        # Moving B away: A and B never share a bin
        adata.obsm["spatial"][labels == "B", 0] += 1000
        table = potential_table(adata, bin_size=20, n_origins=1)
        self.assertTrue(np.isnan(table.loc[("A", "B"), "potential"]))

    def test_skips_missing_genes_and_rejects_normalised_values(self):
        adata = make_tissue()
        st.tl.cci.run_cci_potential(
            adata, ["L_R", "L_MISSING"], use_label="cell_type", verbose=False
        )
        self.assertEqual(set(adata.uns["cci_potential_cell_type"]["lr"]), {"L_R"})
        adata.X = adata.X / 2.5
        with self.assertRaises(ValueError):
            potential_table(adata)


if __name__ == "__main__":
    unittest.main()
