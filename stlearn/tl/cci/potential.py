"""Cell-type-resolved grid communication potential for single-cell spatial data.

A ranking score for every LR and ordered (sender type, receiver type) pair, computed
from raw counts (Xenium, Atera, ...). No p-values.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def run_cci_potential(
    adata: AnnData,
    lrs: npt.NDArray[np.str_],
    use_label: str,
    bin_size: float = 60.0,
    n_origins: int = 4,
    layer: str | None = None,
    library_key: str | None = None,
    spatial_key: str = "spatial",
    verbose: bool = True,
) -> None:
    """Rank LR x (sender type -> receiver type) by grid communication potential.

    Cells are binned into square bins of side bin_size. In each bin, the ligand rate
    of the sender type (pooled ligand counts / pooled library size) is multiplied by
    the receptor rate of the receiver type. Bins are averaged with weight = number of
    ordered sender-receiver cell pairs in the bin. The grid is repeated with n_origins
    origins shifted by bin_size / sqrt(n_origins) and pooled, so the score does not
    depend on where the grid starts. The result is multiplied by the squared median
    library size: an expected L x R count product for a typical pair of cells.

    For sender == receiver only pairs of distinct cells count (a cell's own L x R
    product is removed).

    Parameters
    ----------
    adata: AnnData
        Single-cell spatial data with raw counts in layer (or .X).
    lrs: array of str
        LR pairs as "ligand_receptor"; pairs with a gene missing from adata are
        skipped.
    use_label: str
        Cell type column in adata.obs.
    bin_size: float
        Bin side in the units of the coordinates (um for Xenium): the communication
        scale.
    n_origins: int
        Number of shifted grids, a square number (1, 4, 9, ...). 4 equals sliding
        windows at a stride of half the bin size.
    layer: str | None
        Layer with raw counts; None uses adata.X.
    library_key: str | None
        obs column with library sizes; None uses the total counts of the matrix.
    spatial_key: str
        adata.obsm key with cell centroid coordinates (x, y in the first two columns).
    verbose: bool
        Print progress.

    Returns
    -------
    None. Stores adata.uns[f"cci_potential_{use_label}"], a DataFrame with one row per
    LR x ordered type pair, sorted by potential, with columns lr, ligand, receptor,
    sender, receiver, potential (NaN when the two types never share a bin) and n_eff
    (effective number of bins carrying the score; a low value means one or two
    hotspots drive it).
    """
    # ---- Inputs: raw counts of the LR genes, library sizes, labels, coordinates ----
    lrs = np.asarray(lrs)
    ligands = np.array([lr.split("_")[0] for lr in lrs])
    receptors = np.array([lr.split("_")[1] for lr in lrs])
    present = np.isin(ligands, adata.var_names) & np.isin(receptors, adata.var_names)
    lrs, ligands, receptors = lrs[present], ligands[present], receptors[present]

    matrix = adata.layers[layer] if layer is not None else adata.X
    genes = np.unique(np.concatenate([ligands, receptors]))
    gene_columns = adata.var_names.get_indexer(genes)
    counts = sp.csc_matrix(matrix[:, gene_columns], dtype=np.float64)
    # The rates are count-based; normalised or log values would change every score
    if counts.nnz and not np.allclose(counts.data, np.round(counts.data)):
        raise ValueError("The potential needs raw counts; pass the layer with them.")
    ligand_cols = np.searchsorted(genes, ligands)
    receptor_cols = np.searchsorted(genes, receptors)

    if library_key is None:
        library = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    else:
        library = adata.obs[library_key].to_numpy(dtype=np.float64)
    if not (library > 0).all():
        raise ValueError("Library sizes must be positive; remove empty cells first.")

    labels = adata.obs[use_label].astype("category")
    cell_types = np.array(labels.cat.categories)
    codes = labels.cat.codes.to_numpy()
    n_types = len(cell_types)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    if verbose:
        print(f"{len(lrs)} LRs ({(~present).sum()} skipped), {n_types} cell types.")

    # ---- Pool all shifted grids: numerator sum_b o_b r_b, opportunity sum_b o_b ----
    per_axis = int(round(np.sqrt(n_origins)))
    if per_axis**2 != n_origins:
        raise ValueError("n_origins must be a square number (1, 4, 9, ...).")
    offsets = [bin_size * step / per_axis for step in range(per_axis)]
    same_cell_product = counts[:, ligand_cols].multiply(counts[:, receptor_cols])

    numerator = np.zeros((n_types, n_types, len(lrs)))
    numerator_sq = np.zeros((n_types, n_types, len(lrs)))
    opportunity = np.zeros((n_types, n_types))
    for dy in offsets:
        for dx in offsets:
            origin = coords.min(axis=0) - np.array([dx, dy])
            bin_keys = np.floor((coords - origin) / bin_size).astype(np.int64)
            _, bin_of_cell = np.unique(bin_keys, axis=0, return_inverse=True)
            bin_of_cell = bin_of_cell.ravel()
            n_bins = bin_of_cell.max() + 1

            # Bin x type pseudobulk; same-cell terms are for sender == receiver
            group = bin_of_cell * n_types + codes
            membership = sp.csr_matrix(
                (np.ones(len(group)), (group, np.arange(len(group)))),
                shape=(n_bins * n_types, len(group)),
            )
            summed = (membership @ counts).toarray().reshape(n_bins, n_types, -1)
            own_product = membership @ same_cell_product
            own_product = own_product.toarray().reshape(n_bins, n_types, -1)
            exposure = (membership @ library).reshape(n_bins, n_types)
            exposure_sq = (membership @ library**2).reshape(n_bins, n_types)
            n_cells = np.asarray(membership.sum(axis=1)).reshape(n_bins, n_types)

            for sender in range(n_types):
                ligand_sum = summed[:, sender, ligand_cols]
                for receiver in range(n_types):
                    receptor_sum = summed[:, receiver, receptor_cols]
                    if sender != receiver:
                        pairs = n_cells[:, sender] * n_cells[:, receiver]
                        pair_counts = ligand_sum * receptor_sum
                        pair_exposure = exposure[:, sender] * exposure[:, receiver]
                    else:
                        # Remove the i == j terms: no cell communicates with itself
                        pairs = n_cells[:, sender] * (n_cells[:, sender] - 1)
                        pair_counts = ligand_sum * receptor_sum - own_product[:, sender]
                        sender_exposure = exposure[:, sender]
                        pair_exposure = sender_exposure**2 - exposure_sq[:, sender]
                    has_pairs = pairs > 0
                    # Each bin adds o_b r_b: the pair rate times its number of pairs
                    pair_rate = pair_counts[has_pairs] / pair_exposure[has_pairs, None]
                    bin_numerator = pairs[has_pairs, None] * pair_rate
                    numerator[sender, receiver] += bin_numerator.sum(axis=0)
                    numerator_sq[sender, receiver] += (bin_numerator**2).sum(axis=0)
                    opportunity[sender, receiver] += pairs.sum()

    # ---- Potential = m^2 numerator / opportunity; n_eff = (sum c)^2 / sum c^2 ----
    median_library = np.median(library)
    with np.errstate(invalid="ignore", divide="ignore"):
        potential = median_library**2 * numerator / opportunity[..., None]
        n_eff = numerator**2 / numerator_sq

    sender_index, receiver_index, lr_index = np.indices(potential.shape).reshape(3, -1)
    table = pd.DataFrame(
        {
            "lr": lrs[lr_index],
            "ligand": ligands[lr_index],
            "receptor": receptors[lr_index],
            "sender": cell_types[sender_index],
            "receiver": cell_types[receiver_index],
            "potential": potential.ravel(),
            "n_eff": n_eff.ravel(),
        }
    )
    table = table.sort_values("potential", ascending=False, ignore_index=True)
    adata.uns[f"cci_potential_{use_label}"] = table
    adata.uns[f"cci_potential_{use_label}_params"] = {
        "bin_size": bin_size,
        "n_origins": n_origins,
        "layer": layer,
        "library_key": library_key,
        "median_library_size": median_library,
    }
    if verbose:
        print(f"Stored adata.uns['cci_potential_{use_label}'].")
