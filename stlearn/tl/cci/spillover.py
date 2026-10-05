"""Filtering transcript spillover between touching cells of different types.

In imaging-based data (Xenium, CosMx, MERSCOPE, Atera) segmentation assigns
some transcripts of a cell to its neighbours. A B cell touching a fibroblast
then appears to express a little CXCL12, which in CCI analysis creates
apparent B cell senders right where the real sender sits. On Xenium Rep1 the
fraction of B cells with CXCL12 rises from 31% to 60% with two fibroblasts
within 15 um.

For each gene g and cell i, the expected spillover count is
``lambda_ig = alpha_g * N_ig + beta_g``, where N_ig is the summed count of g in
touching cells of other types, alpha_g is the fraction of a neighbour's
counts that leaks in, and beta_g is the background level. A count is kept
only if it is unlikely under Poisson(lambda_ig).
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import cKDTree
from scipy.stats import poisson

from .single_cell import get_coordinates

_GENE_CHUNK = 64


def _other_type_contacts(
    coords: npt.NDArray[np.float64], codes: npt.NDArray[np.integer], radius: float
) -> sp.csr_matrix:
    """Binary symmetric adjacency between cells of different types within
    radius."""
    n = coords.shape[0]
    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    pairs = pairs[codes[pairs[:, 0]] != codes[pairs[:, 1]]]
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n)
    )


def estimate_spillover(
    counts: sp.csc_matrix,
    contacts: sp.csr_matrix,
    codes: npt.NDArray[np.integer],
    min_type_size: int = 50,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per gene leak fraction alpha and background beta.

    alpha_g is the within-type regression slope of a cell's count on the
    summed count in its touching other-type cells (clipped at 0). Comparing
    cells of the same type removes differences in intrinsic expression
    between types. beta_g is the lowest mean count among cell types with at
    least min_type_size cells, taken as the level a non-expressing type shows.
    """
    n, n_genes = counts.shape
    n_types = int(codes.max()) + 1
    onehot = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, n_types))
    sizes = np.asarray(onehot.sum(axis=0)).ravel()
    big = sizes >= min(min_type_size, sizes.max())
    alpha = np.zeros(n_genes)
    beta = np.zeros(n_genes)
    for start in range(0, n_genes, _GENE_CHUNK):
        cols = slice(start, min(start + _GENE_CHUNK, n_genes))
        x = counts[:, cols].toarray()
        nb = np.asarray(contacts @ x)
        mean_x = (onehot.T @ x) / sizes[:, None]
        mean_nb = (onehot.T @ nb) / sizes[:, None]
        dx = x - mean_x[codes]
        dn = nb - mean_nb[codes]
        denom = (dn**2).sum(axis=0)
        slope = np.divide(
            (dx * dn).sum(axis=0), denom, out=np.zeros_like(denom), where=denom > 0
        )
        alpha[cols] = np.clip(slope, 0.0, None)
        beta[cols] = mean_x[big].min(axis=0)
    return alpha, beta


def filter_spillover(
    adata: AnnData,
    use_label: str,
    radius: float = 15.0,
    pval: float = 0.05,
    layer: str | None = None,
    key_added: str = "spillover_filtered",
    spatial_key: str = "spatial",
    coord_scale: float = 1.0,
    verbose: bool = True,
) -> None:
    """Remove counts explained by spillover from touching cells of other types.

    Run on raw counts, before normalisation, then normalise the stored layer
    and pass it to st.tl.cci.smooth_expression or st.tl.cci.run_sc via layer.

    Each non-zero count x_ig is kept if P(X >= x_ig) < pval under
    Poisson(alpha_g * N_ig + beta_g), with N_ig the summed count of gene g in
    cells of other types within radius of cell i, and alpha_g, beta_g
    estimated from the data (see estimate_spillover); otherwise it is set to
    zero. Counts in cells whose ligand or receptor is truly induced by contact
    with another type look the same as spillover, so some genuine
    contact-dependent expression is removed too; alpha is shared by all cell
    types to limit this.

    Parameters
    ----------
    adata: AnnData
        Cells x genes with raw counts in adata.X or adata.layers[layer].
    use_label: str
        Cell type column in adata.obs.
    radius: float
        Centroid distance within which cells are taken to touch, in
        coordinate units (microns for Xenium/Atera). About one and a half cell
        diameters.
    pval: float
        Counts with a Poisson tail probability at or above this are removed.
        Smaller values remove more.
    layer: str | None
        Layer holding raw counts instead of adata.X.
    key_added: str
        The filtered counts are stored in adata.layers[key_added].
    spatial_key, coord_scale:
        As in st.tl.cci.run_sc.
    verbose: bool
        Print a summary.

    Returns
    -------
    None. Stores adata.layers[key_added] (sparse filtered counts),
    adata.var[f"{key_added}_alpha"], adata.var[f"{key_added}_beta"],
    adata.var[f"{key_added}_frac_removed"] (fraction of non-zero entries
    removed per gene) and adata.uns[f"{key_added}_params"].
    """
    if use_label not in adata.obs:
        raise ValueError(f"{use_label} not found in adata.obs.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    expr = adata.layers[layer] if layer is not None else adata.X
    counts = sp.csc_matrix(expr, dtype=np.float64)
    if counts.nnz and not np.allclose(counts.data, np.round(counts.data)):
        raise ValueError(
            "filter_spillover needs raw counts; run it before normalisation."
        )
    codes = pd.Categorical(adata.obs[use_label]).codes.astype(np.int64)
    coords = get_coordinates(adata, spatial_key, coord_scale)
    contacts = _other_type_contacts(coords, codes, radius)
    alpha, beta = estimate_spillover(counts, contacts, codes)

    keep = np.zeros(counts.nnz, dtype=bool)
    n_genes = counts.shape[1]
    for start in range(0, n_genes, _GENE_CHUNK):
        stop = min(start + _GENE_CHUNK, n_genes)
        nb = np.asarray(contacts @ counts[:, start:stop].toarray())
        for g in range(start, stop):
            lo, hi = counts.indptr[g], counts.indptr[g + 1]
            cells = counts.indices[lo:hi]
            lam = alpha[g] * nb[cells, g - start] + beta[g]
            keep[lo:hi] = poisson.sf(counts.data[lo:hi] - 1, lam) < pval
    filtered = counts.copy()
    filtered.data = np.where(keep, filtered.data, 0.0)
    filtered.eliminate_zeros()

    nnz_per_gene = np.diff(counts.indptr)
    gene_of_entry = np.repeat(np.arange(n_genes), nnz_per_gene)
    kept_per_gene = np.bincount(gene_of_entry[keep], minlength=n_genes)
    frac_removed = np.divide(
        nnz_per_gene - kept_per_gene,
        nnz_per_gene,
        out=np.zeros(n_genes),
        where=nnz_per_gene > 0,
    )
    adata.layers[key_added] = filtered.tocsr()
    adata.var[f"{key_added}_alpha"] = alpha
    adata.var[f"{key_added}_beta"] = beta
    adata.var[f"{key_added}_frac_removed"] = frac_removed
    adata.uns[f"{key_added}_params"] = {
        "use_label": use_label,
        "radius": float(radius),
        "pval": float(pval),
        "layer": layer,
    }
    if verbose:
        n_contacts = np.diff(contacts.indptr)
        print(
            f"Median {int(np.median(n_contacts))} touching other-type cells, "
            f"median leak fraction {np.median(alpha):.3f}; removed "
            f"{1 - keep.mean():.1%} of non-zero counts. Stored in "
            f"adata.layers['{key_added}']."
        )
