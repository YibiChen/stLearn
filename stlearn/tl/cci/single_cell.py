"""Cell-cell interaction analysis at single-cell resolution (Xenium, Atera, etc.).

The Visium pipeline (st.tl.cci.run / st.tl.cci.run_cci) scores ligand-receptor
(LR) co-expression in multi-cell spots and their immediately adjacent spots.
At single-cell resolution three things change:

1. Sender and receiver are different cells. A cell is never its own neighbour,
   and the score is directed: the sender expresses the ligand, the receivers
   around it express the receptor. Optionally, sender and receiver must also be
   of different cell types.
2. Neighbours are every cell whose centroid lies within a user-chosen radius
   (in the units of the coordinates, microns for Xenium/Atera output).
3. Significance is tested in two stages:
   a. Per cell, the LR score is compared with scores of random gene pairs whose
      expression distributions match the ligand and the receptor (the same gene
      permutation idea as st.tl.cci.run).
   b. Per LR, the number of significant sender->receiver edges between each
      pair of cell types is compared with surrogate labellings that keep each
      cell type's composition and spatial autocorrelation (Moran's I), following
      Arthur (2025), doi:10.1111/gean.12417. A plain shuffle, as in
      st.tl.cci.run_cci, is available for comparison.
"""

import os

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from numba import njit, prange
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from . import _gpu
from .perm_utils import gen_rand_pairs
from .spatial_null import permute_labels, permute_values

GRAPH_KEY = "cci_radius_graph"


def _set_threads(n_cpus: int | None) -> None:
    numba.set_num_threads(n_cpus if n_cpus is not None else os.cpu_count())


def get_coordinates(
    adata: AnnData, spatial_key: str = "spatial", coord_scale: float = 1.0
) -> npt.NDArray[np.float64]:
    """Cell centroid coordinates, from adata.obsm[spatial_key] or, failing that,
    adata.obs['imagecol'/'imagerow'], multiplied by coord_scale."""
    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    elif "imagecol" in adata.obs and "imagerow" in adata.obs:
        coords = adata.obs[["imagecol", "imagerow"]].values.astype(np.float64)
    else:
        raise KeyError(
            f"No coordinates found in adata.obsm['{spatial_key}'] or "
            "adata.obs['imagecol'/'imagerow']."
        )
    return coords * coord_scale


def radius_graph(coords: npt.NDArray[np.float64], radius: float) -> sp.csr_matrix:
    """Binary symmetric adjacency linking cells whose centroids are within
    radius of each other. The diagonal is zero, so a cell is never its own
    neighbour."""
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    n = coords.shape[0]
    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    graph.sum_duplicates()
    graph.sort_indices()
    return graph


def _row_normalise(graph: sp.csr_matrix) -> sp.csr_matrix:
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    return sp.csr_matrix(sp.diags(inv) @ graph)


def _gene_features(
    expr_csc: sp.csc_matrix, quantiles: npt.NDArray[np.float64]
) -> npt.NDArray[np.float32]:
    """Per gene: proportion of zeros followed by quantiles of the non-zero
    values. Shape (1 + len(quantiles), n_genes); used to find genes with an
    expression distribution similar to a ligand or receptor."""
    n_cells, n_genes = expr_csc.shape
    feats = np.full((1 + len(quantiles), n_genes), np.nan, dtype=np.float64)
    for g in range(n_genes):
        vals = expr_csc.data[expr_csc.indptr[g] : expr_csc.indptr[g + 1]]
        vals = vals[vals > 0]
        feats[0, g] = 1.0 - len(vals) / n_cells
        if len(vals) > 0:
            feats[1:, g] = np.quantile(vals, quantiles, method="nearest")
    return feats.astype(np.float32)


def _similar_genes(
    ref: npt.NDArray[np.float32],
    n_genes: int,
    cand_feats: npt.NDArray[np.float32],
    cand_names: npt.NDArray[np.str_],
) -> npt.NDArray[np.str_]:
    """The n_genes candidates closest to ref in expression distribution.

    Distance is half the Canberra distance on the zero proportion plus half
    the mean Canberra distance over the non-zero quantiles. Weighting the zero
    proportion this heavily matters for sparse single-cell counts, where the
    non-zero quantiles of many genes are identical.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.abs(cand_feats - ref[:, None]) / (cand_feats + ref[:, None])
    terms = np.nan_to_num(terms, nan=0.0)
    dists = 0.5 * terms[0] + 0.5 * terms[1:].mean(axis=0)
    return cand_names[np.argsort(dists, kind="stable")[:n_genes]]


def _column(expr_csc: sp.csc_matrix, j: int) -> npt.NDArray[np.float64]:
    out = np.zeros(expr_csc.shape[0], dtype=np.float64)
    start, end = expr_csc.indptr[j], expr_csc.indptr[j + 1]
    out[expr_csc.indices[start:end]] = expr_csc.data[start:end]
    return out


def _directed_score(
    sender_expr: npt.NDArray[np.float64],
    neighbour_mean: npt.NDArray[np.float64],
    min_expr: float,
) -> npt.NDArray[np.float64]:
    """Sender expression times the mean partner expression of its neighbours,
    zero unless both exceed min_expr."""
    keep = (sender_expr > min_expr) & (neighbour_mean > min_expr)
    return np.where(keep, sender_expr * neighbour_mean, 0.0)


@njit(parallel=True, cache=True)
def _count_bg_greater(
    obs_scores, cell_idx, lig_bg, rec_bg, pair_l, pair_r, min_expr
):  # pragma: no cover - numba
    """For each tested cell, count random gene pairs scoring >= observed."""
    m = cell_idx.shape[0]
    n_pairs = pair_l.shape[0]
    n_ge = np.zeros(m, dtype=np.int64)
    for t in prange(m):
        i = cell_idx[t]
        s = obs_scores[i]
        c = 0
        for p in range(n_pairs):
            lig = lig_bg[i, pair_l[p]]
            rec = rec_bg[i, pair_r[p]]
            if lig > min_expr and rec > min_expr and lig * rec >= s:
                c += 1
        n_ge[t] = c
    return n_ge


@njit(cache=True)
def _grouped_bh(groups, pvals):  # pragma: no cover - numba
    """Benjamini-Hochberg adjustment applied separately within each group.
    Inputs must be sorted by (group, p-value)."""
    n = pvals.shape[0]
    adj = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start
        while end < n and groups[end] == groups[start]:
            end += 1
        size = end - start
        running = 1.0
        for k in range(end - 1, start - 1, -1):
            rank = k - start + 1
            val = pvals[k] * size / rank
            if val < running:
                running = val
            adj[k] = running
        start = end
    return adj


def _adjust(
    cells: npt.NDArray[np.int64],
    lrs: npt.NDArray[np.int64],
    pvals: npt.NDArray[np.float64],
    correct_axis: str | None,
    adj_method: str,
) -> npt.NDArray[np.float64]:
    if correct_axis is None or len(pvals) == 0:
        return pvals.copy()
    if correct_axis not in ("cell", "LR"):
        raise ValueError("correct_axis must be 'cell', 'LR' or None.")
    groups = cells if correct_axis == "cell" else lrs
    if adj_method != "fdr_bh":
        adj = np.ones_like(pvals)
        for g in np.unique(groups):
            sel = groups == g
            adj[sel] = multipletests(pvals[sel], method=adj_method)[1]
        return adj
    order = np.lexsort((pvals, groups))
    adj_sorted = _grouped_bh(groups[order], pvals[order])
    adj = np.empty_like(pvals)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


def _spatial_global_pval(
    lig_x: npt.NDArray[np.float64],
    rec_x: npt.NDArray[np.float64],
    graph: sp.csr_matrix,
    w_norm: sp.csr_matrix,
    coords: npt.NDArray[np.float64],
    radius: float,
    observed: float,
    n_perms: int,
    min_expr: float,
    seed: int,
) -> float:
    """P-value of the tissue-wide sender score against surrogate L and R that
    keep their expression values and their spatial autocorrelation at the
    cell scale and at coarser grid scales."""
    lig_s, _, _ = permute_values(
        lig_x, graph, n_perms, coords=coords, radius=radius, random_state=seed
    )
    rec_s, _, _ = permute_values(
        rec_x, graph, n_perms, coords=coords, radius=radius, random_state=seed + 1
    )
    rec_mean = np.asarray(w_norm @ rec_s.T)  # cells x perms
    lig_s = lig_s.T
    keep = (lig_s > min_expr) & (rec_mean > min_expr)
    null = np.where(keep, lig_s * rec_mean, 0.0).mean(axis=0)
    return float(((null >= observed).sum() + 1) / (n_perms + 1))


def run_sc(
    adata: AnnData,
    lrs: npt.NDArray[np.str_],
    radius: float,
    spatial_key: str = "spatial",
    coord_scale: float = 1.0,
    use_label: str | None = None,
    different_cell_types: bool = False,
    layer: str | None = None,
    min_cells: int = 10,
    min_expr: float = 0.0,
    n_pairs: int = 1000,
    n_spatial_perms: int = 0,
    adj_method: str = "fdr_bh",
    correct_axis: str | None = "cell",
    pval_adj_cutoff: float = 0.05,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
    device: str | None = None,
    n_cpus: int | None = None,
    random_state: int = 0,
    verbose: bool = True,
) -> None:
    """LR analysis for single-cell resolution spatial data (Xenium, Atera, ...).

    For each LR pair and each cell i, the sender score is
    ``L_i * mean(R_j for j within radius of i, j != i)``, set to zero unless
    both the ligand in i and the mean receptor around i exceed min_expr. The
    receiver score ``R_i * mean(L_j)`` is stored as well. Each cell with a
    sender score is tested against random gene pairs whose expression
    distributions (zero proportion and non-zero quantiles) match the ligand and
    receptor, scored identically at that cell.

    Parameters
    ----------
    adata: AnnData
        Cells x genes, ideally normalised and log-transformed.
    lrs: np.ndarray
        LR pairs to test, in format 'L_R' (see st.tl.cci.load_lrs).
    radius: float
        Search radius around each sender, in coordinate units (microns for
        coordinates from Xenium/Atera cell centroids).
    spatial_key: str
        Key in adata.obsm holding cell centroids; falls back to
        adata.obs['imagecol'/'imagerow'].
    coord_scale: float
        Multiplier converting the coordinates into the units of radius (e.g.
        the pixel size in microns if coordinates are in pixels).
    use_label: str | None
        Cell type column in adata.obs; only needed with different_cell_types.
    different_cell_types: bool
        If True, a sender only counts neighbours of a different cell type.
    layer: str | None
        Expression layer to use instead of adata.X.
    min_cells: int
        Minimum number of cells with a sender score for an LR to be tested.
    min_expr: float
        Expression threshold for a gene to count as expressed.
    n_pairs: int
        Number of random gene pairs per LR.
    n_spatial_perms: int
        If > 0, also test each LR's tissue-wide score against this many
        surrogates in which the ligand and the receptor are each permuted
        across cells while keeping their own Moran's I, on the radius graph and
        on grids of 4x and 16x the radius (Arthur 2025). This
        asks whether L and R co-localise more than two independent genes with
        the same spatial structure would, the concern raised by SOAAR (Khatri
        et al., 2026). Slower; 99 or more is a sensible value.
    adj_method: str
        Multiple testing method, as in statsmodels multipletests.
    correct_axis: str | None
        'cell' corrects across the LRs tested in each cell (as st.tl.cci.run),
        'LR' across the cells tested for each LR, None for no correction.
    pval_adj_cutoff: float
        Adjusted p-value below which a cell is significant for an LR.
    quantiles: tuple
        Non-zero expression quantiles used to match random genes.
    device: str | None
        None (default) counts random gene pairs with numba on the CPU. 'auto',
        'cuda', 'cuda:N' or 'mps' run this step with torch on a GPU ('auto'
        falls back to the CPU if none is found).
    n_cpus: int | None
        Threads to use; all if None.
    random_state: int
        Seed for reproducibility.
    verbose: bool
        Print progress.

    Returns
    -------
    None. Stores:
        adata.obsp['cci_radius_graph']: binary cell adjacency within radius.
        adata.uns['lr_summary']: per LR, n_spots (cells with a sender score),
            n_spots_sig, n_spots_sig_pval, global_score (mean sender score),
            global_pval (versus the random gene pairs), global_padj. Rows are
            sorted by n_spots_sig and give the column order of the obsm results.
            Column names match st.tl.cci.run, with spots meaning cells.
        adata.obsm['lr_scores'], ['lr_receiver_scores'], ['lr_sig_scores'],
            ['-log10(p_vals)'], ['-log10(p_adjs)']: sparse cells x LRs.
        adata.uns['cci_sc_params']: the settings used.
    """
    _set_threads(n_cpus)
    dev = _gpu.resolve_device(device)
    lrs = np.asarray(lrs).astype(str)
    if n_pairs < 100:
        raise ValueError("n_pairs must be >= 100 for a usable background.")
    prob_genes = [g for g in adata.var_names if "_" in g]
    if prob_genes:
        raise ValueError(
            "Gene names containing '_' break the 'L_R' format; rename them: "
            f"{prob_genes[:10]}"
        )

    coords = get_coordinates(adata, spatial_key, coord_scale)
    graph = radius_graph(coords, radius)
    if different_cell_types:
        if use_label is None or use_label not in adata.obs:
            raise ValueError("different_cell_types=True needs use_label in adata.obs.")
        codes = pd.Categorical(adata.obs[use_label]).codes
        coo = graph.tocoo()
        keep = codes[coo.row] != codes[coo.col]
        graph = sp.csr_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.shape
        )
    adata.obsp[GRAPH_KEY] = graph
    w_norm = _row_normalise(graph)
    degrees = np.diff(graph.indptr)
    if verbose:
        print(
            f"Radius {radius}: median {int(np.median(degrees))} neighbours per "
            f"cell, {int((degrees == 0).sum())} cells with none."
        )
    if degrees.sum() == 0:
        raise ValueError("No cell has a neighbour within this radius.")

    expr = adata.layers[layer] if layer is not None else adata.X
    expr_csc = sp.csc_matrix(expr, dtype=np.float64)
    gene_index = {g: i for i, g in enumerate(adata.var_names.astype(str))}

    # Keep LRs whose genes are both measured.
    pairs = [lr.split("_") for lr in lrs]
    present = [(lig in gene_index and rec in gene_index) for lig, rec in pairs]
    n_input = len(lrs)
    lrs = lrs[np.array(present, dtype=bool)] if len(lrs) else lrs
    if verbose:
        # Targeted panels (a few hundred genes) cover few database LR pairs.
        print(f"{len(lrs)} of {n_input} LR pairs have both genes measured.")
    lr_genes = {g for lr in lrs for g in lr.split("_")}

    # Observed scores, filtering LRs with too few sender cells.
    send_cols, recv_cols, kept = [], [], []
    for lr in lrs:
        lig, rec = lr.split("_")
        lig_x = _column(expr_csc, gene_index[lig])
        rec_x = _column(expr_csc, gene_index[rec])
        send = _directed_score(lig_x, w_norm @ rec_x, min_expr)
        if (send > 0).sum() < min_cells:
            continue
        recv = _directed_score(rec_x, w_norm @ lig_x, min_expr)
        send_cols.append(sp.csc_matrix(send[:, None]))
        recv_cols.append(sp.csc_matrix(recv[:, None]))
        kept.append(lr)
    if verbose:
        print(f"{len(kept)} LR pairs with a sender score in >= {min_cells} cells.")
    if len(kept) == 0:
        print("Exiting due to lack of valid LR pairs.")
        return
    lrs = np.array(kept)
    send_scores = sp.hstack(send_cols, format="csc")
    recv_scores = sp.hstack(recv_cols, format="csc")

    # Candidate genes for random pairs: measured, not in any tested LR, with
    # some expression.
    feats = _gene_features(expr_csc, np.asarray(quantiles, dtype=np.float64))
    cand_mask = np.array(
        [g not in lr_genes for g in adata.var_names.astype(str)]
    ) & np.isfinite(feats[1])
    cand_idx = np.where(cand_mask)[0]
    cand_names = np.asarray(adata.var_names, dtype=str)[cand_idx]
    cand_feats = feats[:, cand_idx]
    n_genes = round(np.sqrt(n_pairs) * 2)
    if len(cand_idx) < n_genes:
        raise ValueError(
            f"Need at least {n_genes} non-LR expressed genes to build {n_pairs} "
            f"random pairs, found {len(cand_idx)}."
        )

    n_cells, n_lrs = send_scores.shape
    test_cells, test_lrs, test_pvals = [], [], []
    global_score = np.zeros(n_lrs)
    global_pval = np.ones(n_lrs)
    global_pval_spatial = np.ones(n_lrs)
    similar: dict[str, npt.NDArray[np.str_]] = {}
    with tqdm(
        total=n_lrs,
        desc="Gene-pair permutation per LR",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        disable=not verbose,
    ) as pbar:
        for j, lr in enumerate(lrs):
            lig, rec = lr.split("_")
            for gene in (lig, rec):
                if gene not in similar:
                    ref = feats[:, gene_index[gene]].astype(np.float32)
                    similar[gene] = _similar_genes(ref, n_genes, cand_feats, cand_names)
            l_genes, r_genes = similar[lig], similar[rec]
            rand_pairs = gen_rand_pairs(l_genes, r_genes, n_pairs, random_state + j)
            l_pos = {g: k for k, g in enumerate(l_genes)}
            r_pos = {g: k for k, g in enumerate(r_genes)}
            pair_l = np.array([l_pos[p.split("_")[0]] for p in rand_pairs])
            pair_r = np.array([r_pos[p.split("_")[1]] for p in rand_pairs])

            lig_bg = expr_csc[:, [gene_index[g] for g in l_genes]].toarray()
            rec_bg = np.asarray(
                (w_norm @ expr_csc[:, [gene_index[g] for g in r_genes]]).todense()
            )
            obs = send_scores[:, j].toarray().ravel()
            cells = np.where(obs > 0)[0]
            if dev is None:
                n_ge = _count_bg_greater(
                    obs,
                    cells.astype(np.int64),
                    lig_bg,
                    rec_bg,
                    pair_l,
                    pair_r,
                    min_expr,
                )
            else:
                n_ge = _gpu.count_bg_greater(
                    obs, cells, lig_bg, rec_bg, pair_l, pair_r, min_expr, dev
                )
            test_cells.append(cells)
            test_lrs.append(np.full(len(cells), j))
            test_pvals.append((n_ge + 1) / (n_pairs + 1))

            # Tissue-wide statistic: mean sender score over all cells.
            lig_m = np.where(lig_bg > min_expr, lig_bg, 0.0)
            rec_m = np.where(rec_bg > min_expr, rec_bg, 0.0)
            bg_global = (lig_m.T @ rec_m)[pair_l, pair_r] / n_cells
            global_score[j] = obs.mean()
            global_pval[j] = ((bg_global >= global_score[j]).sum() + 1) / (n_pairs + 1)
            if n_spatial_perms > 0:
                global_pval_spatial[j] = _spatial_global_pval(
                    _column(expr_csc, gene_index[lig]),
                    _column(expr_csc, gene_index[rec]),
                    graph,
                    w_norm,
                    coords,
                    radius,
                    global_score[j],
                    n_spatial_perms,
                    min_expr,
                    random_state + 2 * j,
                )
            pbar.update(1)

    cells_all = np.concatenate(test_cells)
    lrs_all = np.concatenate(test_lrs)
    pvals_all = np.concatenate(test_pvals)
    padj_all = _adjust(cells_all, lrs_all, pvals_all, correct_axis, adj_method)
    sig = padj_all < pval_adj_cutoff

    n_cells_lr = np.bincount(lrs_all, minlength=n_lrs)
    n_sig = np.bincount(lrs_all[sig], minlength=n_lrs)
    n_sig_p = np.bincount(lrs_all[pvals_all < pval_adj_cutoff], minlength=n_lrs)
    global_padj = multipletests(global_pval, method=adj_method)[1]
    spatial_cols = {}
    if n_spatial_perms > 0:
        spatial_cols = {
            "global_pval_spatial": global_pval_spatial,
            "global_padj_spatial": multipletests(
                global_pval_spatial, method=adj_method
            )[1],
        }

    def _sparse(values):
        return sp.csr_matrix(
            (values.astype(np.float32), (cells_all, lrs_all)), shape=(n_cells, n_lrs)
        )

    send_csr = send_scores.tocsr()
    sig_mask = sp.csr_matrix(
        (np.ones(int(sig.sum()), dtype=np.float32), (cells_all[sig], lrs_all[sig])),
        shape=(n_cells, n_lrs),
    )
    sig_scores = sp.csr_matrix(send_csr.multiply(sig_mask), dtype=np.float32)
    order = np.argsort(-n_sig, kind="stable")
    summary = pd.DataFrame(
        {
            "n_spots": n_cells_lr,
            "n_spots_sig": n_sig,
            "n_spots_sig_pval": n_sig_p,
            "global_score": global_score,
            "global_pval": global_pval,
            "global_padj": global_padj,
            **spatial_cols,
        },
        index=lrs,
    ).iloc[order]

    adata.uns["lr_summary"] = summary
    adata.obsm["lr_scores"] = send_csr[:, order].astype(np.float32)
    adata.obsm["lr_receiver_scores"] = recv_scores.tocsr()[:, order].astype(np.float32)
    adata.obsm["lr_sig_scores"] = sig_scores[:, order]
    adata.obsm["-log10(p_vals)"] = _sparse(-np.log10(pvals_all))[:, order]
    adata.obsm["-log10(p_adjs)"] = _sparse(-np.log10(padj_all))[:, order]
    adata.uns["cci_sc_params"] = {
        "radius": float(radius),
        "spatial_key": spatial_key,
        "coord_scale": float(coord_scale),
        "use_label": use_label,
        "different_cell_types": bool(different_cell_types),
        "layer": layer,
        "min_expr": float(min_expr),
        "n_pairs": int(n_pairs),
        "n_spatial_perms": int(n_spatial_perms),
        "correct_axis": correct_axis,
        "adj_method": adj_method,
        "pval_adj_cutoff": float(pval_adj_cutoff),
    }
    if verbose:
        print(
            "Stored adata.uns['lr_summary'], adata.obsp['cci_radius_graph'] and "
            "sparse cells x LR matrices in adata.obsm: 'lr_scores', "
            "'lr_receiver_scores', 'lr_sig_scores', '-log10(p_vals)', "
            "'-log10(p_adjs)'. obsm columns follow the rows of lr_summary."
        )


@njit(parallel=True, cache=True)
def _null_counts_ge(src, dst, perms, observed, n_types):  # pragma: no cover - numba
    """Count surrogate labellings whose sender->receiver type counts are >= the
    observed counts, and sum the surrogate counts (for the null mean)."""
    n_perms = perms.shape[0]
    ge = np.zeros((n_perms, n_types, n_types), dtype=np.int64)
    tot = np.zeros((n_perms, n_types, n_types), dtype=np.int64)
    for p in prange(n_perms):
        counts = np.zeros((n_types, n_types), dtype=np.int64)
        lab = perms[p]
        for e in range(src.shape[0]):
            counts[lab[src[e]], lab[dst[e]]] += 1
        for a in range(n_types):
            for b in range(n_types):
                if counts[a, b] >= observed[a, b]:
                    ge[p, a, b] = 1
        tot[p] = counts
    out = np.zeros((n_types, n_types), dtype=np.int64)
    total = np.zeros((n_types, n_types), dtype=np.int64)
    for p in range(n_perms):
        out += ge[p]
        total += tot[p]
    return out, total


def run_cci_sc(
    adata: AnnData,
    use_label: str,
    n_perms: int = 100,
    null: str = "spatial",
    min_spots: int = 3,
    sig_spots: bool = True,
    different_cell_types: bool = False,
    p_cutoff: float = 0.05,
    adj_method: str | None = None,
    eps: float = 0.01,
    init: str = "smooth",
    max_proposals_per_cell: int = 200,
    device: str | None = None,
    n_cpus: int | None = None,
    random_state: int = 0,
    verbose: bool = True,
) -> None:
    """Cell type to cell type interactions per LR at single-cell resolution.

    Run st.tl.cci.run_sc first. For each LR, every significant sender cell i
    (ligand) and every receiver j within the radius (receptor expressed,
    j != i) forms a directed edge i -> j. Edges are counted per (sender type,
    receiver type) and compared with surrogate labellings. With null='spatial'
    the surrogates keep the composition and each cell type's Moran's I on the
    radius graph (Arthur 2025), so cell types that sit in large patches are not
    called interacting just because they cluster. The same surrogates are
    reused for every LR.

    Both nulls move labels away from expression, so a pair is significant
    whenever its sender type expresses L and its receiver type R and the two
    are adjacent; with cell type specific genes most such pairs pass. Rank
    them by the stored enrichment. In imaging data, transcripts from one cell
    are often assigned to the touching cell (segmentation spillover), so
    low-level L or R in a cell next to a high expresser of another type can
    create apparent senders or receivers of that type.

    Parameters
    ----------
    adata: AnnData
        Must have had st.tl.cci.run_sc run.
    use_label: str
        Cell type column in adata.obs.
    n_perms: int
        Number of surrogate labellings; p-values have resolution 1/(n_perms+1).
    null: str
        'spatial' (autocorrelation-preserving) or 'random' (plain shuffle).
    min_spots: int
        Minimum significant sender cells (or scored cells if not sig_spots) for
        an LR to be tested.
    sig_spots: bool
        Use only senders significant in run_sc; else every scored sender.
    different_cell_types: bool
        Ignore same-type edges (diagonal set to count 0, p-value 1).
    p_cutoff: float
        Significance cutoff for cell type pairs.
    adj_method: str | None
        If given, adjust p-values across cell type pairs within each LR with
        statsmodels multipletests; None uses raw p-values as st.tl.cci.run_cci.
    eps, init, max_proposals_per_cell:
        Passed to the spatial resampler, see
        stlearn.tl.cci.spatial_null.permute_labels.
    device: str | None
        None (default) counts edges under each surrogate with numba on the
        CPU; 'auto', 'cuda', 'cuda:N' or 'mps' use torch on a GPU. Generating
        the spatial surrogates is a sequential swap chain and stays on the CPU
        (parallel across surrogates).
    n_cpus: int | None
        Threads to use; all if None.
    random_state: int
        Seed.
    verbose: bool
        Print progress.

    Returns
    -------
    None. Stores the same keys as st.tl.cci.run_cci, with matrices directed
    (rows are sender types, columns receiver types):
        adata.uns['lr_summary'] columns f"n_cci_sig_{use_label}",
            f"n-spot_cci_{use_label}", f"n-spot_cci_sig_{use_label}" (edge
            counts).
        adata.uns[f"lr_cci_{use_label}"], [f"lr_cci_raw_{use_label}"]
        adata.uns[f"per_lr_cci_{use_label}"], [f"per_lr_cci_pvals_{use_label}"],
            [f"per_lr_cci_raw_{use_label}"]
        adata.uns[f"per_lr_cci_enrichment_{use_label}"]: per LR, observed edge
            count over the mean surrogate count, (obs + 1) / (null mean + 1).
            On large tissues most tested pairs reach the minimum p-value, so
            use this to rank significant pairs.
        adata.uns[f"cci_sc_null_{use_label}"]: observed and surrogate Moran's I
            per cell type, to check the null reproduced the clustering.
    """
    _set_threads(n_cpus)
    dev = _gpu.resolve_device(device)
    if "lr_summary" not in adata.uns or GRAPH_KEY not in adata.obsp:
        raise ValueError("Run st.tl.cci.run_sc first.")
    if use_label not in adata.obs:
        raise ValueError(f"{use_label} not found in adata.obs.")
    params = adata.uns.get("cci_sc_params", {})
    min_expr = float(params.get("min_expr", 0.0))
    layer = params.get("layer", None)

    cats = pd.Categorical(adata.obs[use_label].astype(str))
    codes = cats.codes.astype(np.int32)
    all_set = np.array(cats.categories, dtype=str)
    n_types = len(all_set)

    # The resampler uses the plain radius graph (both directions, all types).
    if params.get("different_cell_types", False):
        coords = get_coordinates(
            adata, params.get("spatial_key", "spatial"), params.get("coord_scale", 1.0)
        )
        null_graph = radius_graph(coords, params["radius"])
    else:
        null_graph = adata.obsp[GRAPH_KEY]
    null_graph = sp.csr_matrix(null_graph)

    if verbose:
        print(f"Generating {n_perms} {null} surrogate labellings of {use_label}...")
    perms, achieved, observed_i = permute_labels(
        codes,
        null_graph,
        n_perms,
        method=null,
        eps=eps,
        init=init,
        max_proposals_per_cell=max_proposals_per_cell,
        random_state=random_state,
    )
    if null == "spatial":
        gap = np.abs(achieved - observed_i).max(axis=0)
        bad = all_set[gap > 5 * eps]
        if len(bad) > 0:
            print(
                "Warning: surrogates did not reach the observed Moran's I for "
                f"{list(bad)}; consider a larger max_proposals_per_cell."
            )

    counter = None if dev is None else _gpu.EdgeCounter(perms, n_types, dev)
    graph = sp.csr_matrix(adata.obsp[GRAPH_KEY])
    expr = adata.layers[layer] if layer is not None else adata.X
    expr_csc = sp.csc_matrix(expr, dtype=np.float64)
    gene_index = {g: i for i, g in enumerate(adata.var_names.astype(str))}

    lr_summary = adata.uns["lr_summary"]
    lrs = lr_summary.index.values.astype(str)
    score_key = "lr_sig_scores" if sig_spots else "lr_scores"
    scores = sp.csc_matrix(adata.obsm[score_key])
    n_senders = np.diff(scores.indptr)
    test_idx = np.where(n_senders > min_spots)[0]
    if len(test_idx) == 0:
        raise ValueError(
            "No LR pairs pass min_spots; relax min_spots, sig_spots or run_sc."
        )

    all_matrix = np.zeros((n_types, n_types), dtype=np.int64)
    raw_matrix = np.zeros((n_types, n_types), dtype=np.int64)
    per_lr_cci, per_lr_cci_pvals, per_lr_cci_raw = {}, {}, {}
    per_lr_cci_enrich = {}
    n_edges = np.zeros(len(lrs))
    n_edges_sig = np.zeros(len(lrs))
    n_cci_sig = np.zeros(len(lrs))
    off_diag = ~np.eye(n_types, dtype=bool)
    for j in tqdm(test_idx, desc="Cell type permutation per LR", disable=not verbose):
        lr = lrs[j]
        _, rec = lr.split("_")
        senders = scores.indices[scores.indptr[j] : scores.indptr[j + 1]]
        rec_on = _column(expr_csc, gene_index[rec]) > min_expr
        sub = graph[senders].tocoo()
        keep = rec_on[sub.col]
        src = senders[sub.row[keep]].astype(np.int64)
        dst = sub.col[keep].astype(np.int64)

        observed = np.zeros((n_types, n_types), dtype=np.int64)
        np.add.at(observed, (codes[src], codes[dst]), 1)
        if different_cell_types:
            np.fill_diagonal(observed, 0)
        if n_perms > 0 and len(src) > 0:
            if counter is None:
                n_ge, null_sum = _null_counts_ge(src, dst, perms, observed, n_types)
            else:
                n_ge, null_sum = counter.ge_counts(src, dst, observed)
            pvals = (n_ge + 1) / (n_perms + 1)
            null_mean = null_sum / n_perms
        else:
            pvals = np.ones((n_types, n_types))
            null_mean = np.zeros((n_types, n_types))
        # With 1e5+ cells most tested pairs reach the smallest attainable
        # p-value, so the enrichment over the null is what ranks them.
        enrichment = (observed + 1) / (null_mean + 1)
        pvals[observed == 0] = 1.0
        if different_cell_types:
            np.fill_diagonal(pvals, 1.0)
        if adj_method is not None:
            mask = off_diag if different_cell_types else np.ones_like(off_diag)
            pvals[mask] = multipletests(pvals[mask], method=adj_method)[1]

        sig_matrix = observed.copy()
        sig_matrix[pvals >= p_cutoff] = 0
        n_edges[j] = observed.sum()
        n_edges_sig[j] = sig_matrix.sum()
        n_cci_sig[j] = (sig_matrix > 0).sum()
        raw_matrix += observed
        all_matrix += sig_matrix
        per_lr_cci[lr] = pd.DataFrame(sig_matrix, index=all_set, columns=all_set)
        per_lr_cci_pvals[lr] = pd.DataFrame(pvals, index=all_set, columns=all_set)
        per_lr_cci_raw[lr] = pd.DataFrame(observed, index=all_set, columns=all_set)
        per_lr_cci_enrich[lr] = pd.DataFrame(enrichment, index=all_set, columns=all_set)

    lr_summary[f"n_cci_sig_{use_label}"] = n_cci_sig
    lr_summary[f"n-spot_cci_{use_label}"] = n_edges
    lr_summary[f"n-spot_cci_sig_{use_label}"] = n_edges_sig
    adata.uns["lr_summary"] = lr_summary
    adata.uns[f"lr_cci_{use_label}"] = pd.DataFrame(
        all_matrix, index=all_set, columns=all_set
    )
    adata.uns[f"lr_cci_raw_{use_label}"] = pd.DataFrame(
        raw_matrix, index=all_set, columns=all_set
    )
    adata.uns[f"per_lr_cci_{use_label}"] = per_lr_cci
    adata.uns[f"per_lr_cci_pvals_{use_label}"] = per_lr_cci_pvals
    adata.uns[f"per_lr_cci_raw_{use_label}"] = per_lr_cci_raw
    adata.uns[f"per_lr_cci_enrichment_{use_label}"] = per_lr_cci_enrich
    adata.uns[f"cci_sc_null_{use_label}"] = pd.DataFrame(
        {
            "observed_morans_i": observed_i,
            "null_mean_morans_i": achieved.mean(axis=0),
            "null_max_abs_gap": np.abs(achieved - observed_i).max(axis=0),
        },
        index=all_set,
    )
    if verbose:
        print(
            f"Stored directed (sender x receiver) results in "
            f"adata.uns['per_lr_cci_{use_label}'] and related keys."
        )


def smooth_expression(
    adata: AnnData,
    use_label: str,
    radius: float,
    sigma: float | None = None,
    self_weight: float = 1.0,
    spatial_key: str = "spatial",
    coord_scale: float = 1.0,
    layer: str | None = None,
    key_added: str = "smoothed",
    verbose: bool = True,
) -> None:
    """Smooth expression over nearby cells of the same type to reduce dropout.

    Each cell's expression becomes a weighted average of itself and the cells
    of the same type within radius, with Gaussian weights
    ``exp(-d^2 / (2 sigma^2))`` on the centroid distance d. The cell itself
    gets self_weight (1.0 is the kernel value at d = 0). Restricting the
    average to the same cell type keeps a ligand made by one type from being
    spread onto neighbouring cells of another type, so sender and receiver
    identities in run_sc stay intact.

    Use the result with ``st.tl.cci.run_sc(..., layer=key_added)``. The random
    gene pairs are then drawn from the same smoothed layer, so the per-cell
    test compares like with like. Smoothing does raise spatial
    autocorrelation, so keep the radius small (about one to two cell
    diameters) and prefer the spatial nulls when testing.

    Parameters
    ----------
    adata: AnnData
        Cells x genes.
    use_label: str
        Cell type column in adata.obs.
    radius: float
        Neighbourhood radius in coordinate units (microns for Xenium/Atera).
    sigma: float | None
        Gaussian kernel width; defaults to radius / 2.
    self_weight: float
        Weight of the cell's own expression before row normalisation.
    spatial_key, coord_scale:
        As in run_sc.
    layer: str | None
        Expression layer to smooth instead of adata.X.
    key_added: str
        Smoothed matrix is stored in adata.layers[key_added].
    verbose: bool
        Print a summary.
    """
    if use_label not in adata.obs:
        raise ValueError(f"{use_label} not found in adata.obs.")
    sigma = radius / 2 if sigma is None else sigma
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    coords = get_coordinates(adata, spatial_key, coord_scale)
    n = coords.shape[0]
    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    codes = pd.Categorical(adata.obs[use_label]).codes
    pairs = pairs[codes[pairs[:, 0]] == codes[pairs[:, 1]]]
    dist2 = ((coords[pairs[:, 0]] - coords[pairs[:, 1]]) ** 2).sum(axis=1)
    w = np.exp(-dist2 / (2 * sigma**2))
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
    vals = np.concatenate([w, w, np.full(n, self_weight)])
    weights = _row_normalise(sp.csr_matrix((vals, (rows, cols)), shape=(n, n)))

    expr = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(expr):
        smoothed = sp.csr_matrix(weights @ sp.csr_matrix(expr))
    else:
        smoothed = weights @ np.asarray(expr)
    adata.layers[key_added] = smoothed
    adata.uns[f"{key_added}_params"] = {
        "use_label": use_label,
        "radius": float(radius),
        "sigma": float(sigma),
        "self_weight": float(self_weight),
        "layer": layer,
    }
    if verbose:
        n_neigh = np.diff(weights.indptr) - 1
        print(
            f"Smoothed over a median of {int(np.median(n_neigh))} same-type "
            f"neighbours; stored in adata.layers['{key_added}']."
        )
