"""Spatially-constrained null models for cell type labels.

Randomly shuffling cell type labels across cells (complete spatial randomness)
destroys the spatial clustering that cell types naturally have in tissue. At
single-cell resolution this inflates the significance of interactions between
cell types that sit in large contiguous patches, because the null almost never
reproduces those patches.

The resampler here follows Arthur (2025), "A General Method for Resampling
Autocorrelated Spatial Data", Geographical Analysis, doi:10.1111/gean.12417
(arXiv:2401.05728). Starting from a random permutation of the labels, pairs of
cells swap labels, accepting a swap only when it moves each cell type's Moran's I
closer to the value observed in the real tissue (zero-temperature
Metropolis-Hastings). Swaps keep the cell type composition exact. Arthur shows
the target should be approached from above to get correct false positive
rates, and that a random start struggles when Moran's I is above about 0.7,
which is common for cell types at single-cell resolution. By default we
therefore start from random, spatially coherent patches (smoothed noise, exact
composition) whose autocorrelation exceeds the target, then swap down to it.
Arthur's own random start, with or without pre-freezing, is available too.

For a categorical label we track one indicator per cell type and minimise
E = sum_k (I_k_target - I_k)^2, where I_k is Moran's I of the indicator of
type k on the binary radius graph.
"""

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from numba import njit, prange


@njit(cache=True)
def _neighbour_type_counts(labels, indptr, indices, n_types):
    """Count, per cell, the neighbours belonging to each cell type."""
    n = labels.shape[0]
    counts = np.zeros((n, n_types), dtype=np.int32)
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            counts[i, labels[indices[p]]] += 1
    return counts


@njit(cache=True)
def _morans_from_counts(labels, counts, degrees, type_sizes, s0):
    """Moran's I of each cell type indicator given neighbour type counts."""
    n = labels.shape[0]
    n_types = type_sizes.shape[0]
    numer = np.zeros(n_types, dtype=np.float64)
    for i in range(n):
        for k in range(n_types):
            m_k = type_sizes[k] / n
            z_i = (1.0 if labels[i] == k else 0.0) - m_k
            lag_i = counts[i, k] - m_k * degrees[i]
            numer[k] += z_i * lag_i
    morans = np.zeros(n_types, dtype=np.float64)
    for k in range(n_types):
        denom = type_sizes[k] - type_sizes[k] * type_sizes[k] / n
        if denom > 0 and s0 > 0:
            morans[k] = (n / s0) * numer[k] / denom
    return numer, morans


@njit(cache=True)
def _is_neighbour(a, b, indptr, indices):
    for p in range(indptr[a], indptr[a + 1]):
        if indices[p] == b:
            return 1.0
    return 0.0


@njit(cache=True)
def _greedy_phase(
    labels,
    counts,
    numer,
    morans,
    targets,
    indptr,
    indices,
    degrees,
    type_sizes,
    s0,
    eps,
    max_proposals,
    stall_limit,
):
    """Swap labels until every |I_k - target_k| < eps, or we stall / run out.

    Modifies labels, counts, numer and morans in place.
    """
    n = labels.shape[0]
    n_types = type_sizes.shape[0]
    scale = np.zeros(n_types, dtype=np.float64)
    for k in range(n_types):
        denom = type_sizes[k] - type_sizes[k] * type_sizes[k] / n
        if denom > 0 and s0 > 0:
            scale[k] = (n / s0) / denom

    rejections = 0
    for proposal in range(max_proposals):
        if proposal % 64 == 0:
            worst = 0.0
            for k in range(n_types):
                dev = abs(targets[k] - morans[k])
                if dev > worst:
                    worst = dev
            if worst < eps:
                break

        a = np.random.randint(n)  # noqa: NPY002 (numba requires legacy API)
        b = np.random.randint(n)  # noqa: NPY002
        ta = labels[a]
        tb = labels[b]
        if ta == tb:
            continue

        w_ab = _is_neighbour(a, b, indptr, indices)
        m_a = type_sizes[ta] / n
        m_b = type_sizes[tb] / n
        # Spatial lags of the centred indicators at a and b.
        lag_ta_a = counts[a, ta] - m_a * degrees[a]
        lag_ta_b = counts[b, ta] - m_a * degrees[b]
        lag_tb_a = counts[a, tb] - m_b * degrees[a]
        lag_tb_b = counts[b, tb] - m_b * degrees[b]
        # Swapping values z_a <-> z_b changes sum_ij w_ij z_i z_j by
        # 2*d*(lag_a - lag_b) - 2*w_ab*d^2, with d = z_b - z_a.
        d_numer_ta = -2.0 * (lag_ta_a - lag_ta_b) - 2.0 * w_ab
        d_numer_tb = 2.0 * (lag_tb_a - lag_tb_b) - 2.0 * w_ab
        new_i_ta = (numer[ta] + d_numer_ta) * scale[ta]
        new_i_tb = (numer[tb] + d_numer_tb) * scale[tb]
        d_energy = (
            (targets[ta] - new_i_ta) ** 2
            - (targets[ta] - morans[ta]) ** 2
            + (targets[tb] - new_i_tb) ** 2
            - (targets[tb] - morans[tb]) ** 2
        )
        if d_energy <= 0.0:
            labels[a] = tb
            labels[b] = ta
            for p in range(indptr[a], indptr[a + 1]):
                j = indices[p]
                counts[j, ta] -= 1
                counts[j, tb] += 1
            for p in range(indptr[b], indptr[b + 1]):
                j = indices[p]
                counts[j, tb] -= 1
                counts[j, ta] += 1
            numer[ta] += d_numer_ta
            numer[tb] += d_numer_tb
            morans[ta] = new_i_ta
            morans[tb] = new_i_tb
            rejections = 0
        else:
            rejections += 1
            if rejections > stall_limit:
                break


@njit(cache=True)
def _smooth_field(field, indptr, indices, degrees, n_steps):
    """Lazy neighbour averaging of each column of field, in place."""
    n, n_types = field.shape
    smoothed = np.empty_like(field)
    for _ in range(n_steps):
        for i in range(n):
            for k in range(n_types):
                smoothed[i, k] = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                for k in range(n_types):
                    smoothed[i, k] += field[j, k]
            for k in range(n_types):
                if degrees[i] > 0:
                    smoothed[i, k] = (
                        0.5 * field[i, k] + 0.5 * smoothed[i, k] / degrees[i]
                    )
                else:
                    smoothed[i, k] = field[i, k]
        field[:, :] = smoothed


@njit(cache=True)
def _labels_from_field(field, type_sizes):
    """Fill cells from the highest standardised (cell, type) field values while
    respecting each type's count, giving patches with exact composition."""
    n, n_types = field.shape
    scaled = np.empty_like(field)
    for k in range(n_types):
        col = field[:, k]
        sd = col.std()
        scaled[:, k] = (col - col.mean()) / (sd if sd > 0 else 1.0)
    order = np.argsort(-scaled.ravel())
    labels = np.full(n, -1, dtype=np.int32)
    quota = np.empty(n_types, dtype=np.int64)
    for k in range(n_types):
        quota[k] = np.int64(type_sizes[k])
    n_left = n
    for idx in order:
        cell = idx // n_types
        k = idx % n_types
        if labels[cell] < 0 and quota[k] > 0:
            labels[cell] = k
            quota[k] -= 1
            n_left -= 1
            if n_left == 0:
                break
    return labels


@njit(cache=True)
def _moran_resample_one(
    labels_obs,
    indptr,
    indices,
    type_sizes,
    targets,
    init,
    eps,
    max_proposals,
    stall_limit,
    seed,
):
    """One surrogate labelling preserving composition and per-type Moran's I.

    init: 0 = random permutation then greedy (Arthur 2025, Alg. 1);
          1 = random permutation, pre-freeze to 2x target, then greedy (Alg. 2);
          2 = smooth random patches above the target, then greedy down to it.
    """
    np.random.seed(seed)  # noqa: NPY002 (numba requires legacy API)
    n = labels_obs.shape[0]
    n_types = type_sizes.shape[0]
    degrees = np.zeros(n, dtype=np.float64)
    for i in range(n):
        degrees[i] = indptr[i + 1] - indptr[i]
    s0 = degrees.sum()

    if init == 2:
        # Smooth more until every type starts at or above its target, so the
        # greedy phase only has to break structure, which converges quickly.
        field = np.random.standard_normal((n, n_types))  # noqa: NPY002
        done_steps = 0
        n_steps = 4
        while True:
            _smooth_field(field, indptr, indices, degrees, n_steps - done_steps)
            done_steps = n_steps
            labels = _labels_from_field(field, type_sizes)
            counts = _neighbour_type_counts(labels, indptr, indices, n_types)
            numer, morans = _morans_from_counts(labels, counts, degrees, type_sizes, s0)
            below = False
            for k in range(n_types):
                if morans[k] < targets[k]:
                    below = True
            if not below or n_steps >= 4096:
                break
            n_steps *= 2
    else:
        labels = np.random.permutation(labels_obs)  # noqa: NPY002
        counts = _neighbour_type_counts(labels, indptr, indices, n_types)
        numer, morans = _morans_from_counts(labels, counts, degrees, type_sizes, s0)
        if init == 1:
            high = np.empty(n_types, dtype=np.float64)
            for k in range(n_types):
                high[k] = 2.0 * targets[k]
            _greedy_phase(
                labels,
                counts,
                numer,
                morans,
                high,
                indptr,
                indices,
                degrees,
                type_sizes,
                s0,
                eps,
                max_proposals,
                stall_limit,
            )

    _greedy_phase(
        labels,
        counts,
        numer,
        morans,
        targets,
        indptr,
        indices,
        degrees,
        type_sizes,
        s0,
        eps,
        max_proposals,
        stall_limit,
    )
    return labels, morans


@njit(parallel=True, cache=True)
def _moran_resample_many(
    labels_obs,
    indptr,
    indices,
    type_sizes,
    targets,
    n_perms,
    init,
    eps,
    max_proposals,
    stall_limit,
    seed,
):
    n = labels_obs.shape[0]
    out = np.zeros((n_perms, n), dtype=np.int32)
    achieved = np.zeros((n_perms, type_sizes.shape[0]), dtype=np.float64)
    for p in prange(n_perms):
        # Seeding per surrogate makes each one depend only on (seed + p), so
        # results do not depend on the number of threads.
        labels, morans = _moran_resample_one(
            labels_obs,
            indptr,
            indices,
            type_sizes,
            targets,
            init,
            eps,
            max_proposals,
            stall_limit,
            seed + p,
        )
        out[p, :] = labels
        achieved[p, :] = morans
    return out, achieved


def label_morans_i(
    codes: npt.NDArray[np.integer], graph: sp.csr_matrix, n_types: int
) -> npt.NDArray[np.float64]:
    """Moran's I of each cell type indicator on a binary symmetric graph.

    Parameters
    ----------
    codes:
        Integer cell type code per cell, in 0..n_types-1.
    graph:
        Binary, symmetric adjacency (cells x cells) with zero diagonal.
    n_types:
        Number of cell types.
    """
    graph = sp.csr_matrix(graph)
    codes = np.asarray(codes, dtype=np.int32)
    indptr = graph.indptr.astype(np.int64)
    indices = graph.indices.astype(np.int64)
    degrees = np.diff(indptr).astype(np.float64)
    type_sizes = np.bincount(codes, minlength=n_types).astype(np.float64)
    counts = _neighbour_type_counts(codes, indptr, indices, n_types)
    _, morans = _morans_from_counts(codes, counts, degrees, type_sizes, degrees.sum())
    return morans


def permute_labels(
    codes: npt.NDArray[np.integer],
    graph: sp.csr_matrix,
    n_perms: int,
    method: str = "spatial",
    eps: float = 0.01,
    init: str = "smooth",
    max_proposals_per_cell: int = 200,
    stall_per_cell: int = 10,
    random_state: int = 0,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate surrogate cell type labellings for the cell type permutation test.

    Parameters
    ----------
    codes:
        Integer cell type code per cell.
    graph:
        Binary symmetric radius graph (cells x cells), zero diagonal; defines the
        spatial weights for Moran's I.
    n_perms:
        Number of surrogate labellings.
    method:
        'spatial' preserves each cell type's Moran's I (Arthur 2025);
        'random' is a plain shuffle (complete spatial randomness), as used by
        st.tl.cci.run_cci.
    eps:
        Tolerance on |I_surrogate - I_observed| per cell type.
    init:
        Starting point for the greedy swaps when method='spatial'. 'smooth'
        (default) starts from random spatially coherent patches whose
        autocorrelation is above the target, so the target is approached from
        above; this reaches the strong clustering typical of cell types at
        single-cell resolution (Moran's I > 0.7), where a random start gets stuck.
        'prefreeze' is Arthur's Algorithm 2 (random start, greedy up to twice the
        target, then down). 'random' is Algorithm 1 (random start, greedy to
        target).
    max_proposals_per_cell:
        Cap on swap proposals per phase, as a multiple of the number of cells.
    stall_per_cell:
        Stop a phase after this many consecutive rejected swaps, as a multiple
        of the number of cells.
    random_state:
        Seed; surrogate p uses random_state + p.

    Returns
    -------
    perms:
        (n_perms, n_cells) surrogate codes.
    achieved:
        (n_perms, n_types) Moran's I reached by each surrogate.
    observed:
        (n_types,) Moran's I of the observed labels.
    """
    codes = np.asarray(codes, dtype=np.int32)
    n_types = int(codes.max()) + 1
    observed = label_morans_i(codes, graph, n_types)

    if method == "random":
        rng = np.random.default_rng(random_state)
        perms = np.vstack([rng.permutation(codes) for _ in range(n_perms)]).astype(
            np.int32
        )
        achieved = np.vstack([label_morans_i(p, graph, n_types) for p in perms])
        return perms, achieved, observed
    if method != "spatial":
        raise ValueError(f"method must be 'spatial' or 'random', got {method!r}.")

    inits = {"random": 0, "prefreeze": 1, "smooth": 2}
    if init not in inits:
        raise ValueError(f"init must be one of {list(inits)}, got {init!r}.")
    graph = sp.csr_matrix(graph)
    n = len(codes)
    type_sizes = np.bincount(codes, minlength=n_types).astype(np.float64)
    perms, achieved = _moran_resample_many(
        codes,
        graph.indptr.astype(np.int64),
        graph.indices.astype(np.int64),
        type_sizes,
        observed.astype(np.float64),
        int(n_perms),
        inits[init],
        float(eps),
        int(max_proposals_per_cell * n),
        int(stall_per_cell * n),
        int(random_state),
    )
    return perms, achieved, observed


# --- Continuous values (gene expression) ------------------------------------
#
# A single Moran's I at the cell-neighbour scale does not pin down the size of
# expression domains: for sparse single-cell counts it is dominated by dropout
# noise, and many small blobs can match it as well as one large domain. Gene
# surrogates therefore also match Moran's I of bin-averaged expression on coarser
# square grids (by default bins of 4x and 16x the radius; a scale with fewer
# than MIN_BINS occupied bins is skipped). Each swap changes at most two bins per
# scale, so every scale updates in constant time.


MIN_BINS = 30


def grid_structure(
    coords: npt.NDArray[np.float64], bin_size: float
) -> tuple[npt.NDArray[np.int64], sp.csr_matrix]:
    """Assign cells to square bins and link non-empty bins to their (up to 8)
    neighbours. Returns bin_of_cell and the binary bin adjacency."""
    ij = np.floor((coords - coords.min(axis=0)) / bin_size).astype(np.int64)
    width = int(ij[:, 1].max()) + 3
    keys = ij[:, 0] * width + ij[:, 1]
    uniq, bin_of_cell = np.unique(keys, return_inverse=True)
    lookup = {int(k): b for b, k in enumerate(uniq)}
    rows, cols = [], []
    for b, k in enumerate(uniq):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nb = lookup.get(int(k) + di * width + dj)
                if nb is not None and nb != b:
                    rows.append(b)
                    cols.append(nb)
    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(uniq), len(uniq))
    )
    adj.sort_indices()
    return bin_of_cell.astype(np.int64).ravel(), adj


@njit(cache=True)
def _values_moran_state(values, indptr, indices):
    n = values.shape[0]
    mean = values.mean()
    lags = np.zeros(n, dtype=np.float64)
    numer = 0.0
    denom = 0.0
    s0 = 0.0
    for i in range(n):
        z_i = values[i] - mean
        denom += z_i * z_i
        for p in range(indptr[i], indptr[i + 1]):
            lags[i] += values[indices[p]] - mean
        s0 += indptr[i + 1] - indptr[i]
        numer += z_i * lags[i]
    scale = (n / s0) / denom if denom > 0 and s0 > 0 else 0.0
    return lags, numer, scale


@njit(cache=True)
def _grid_state(values, bin_of_cell, b_indptr, b_indices, n_bins):
    """Bin means, bin lags and the running sums giving Moran's I of bin means.

    stats = [A, B, C, D, S0, G] with A = sum_g m_g lag_g, B = sum_g m_g deg_g,
    C = sum_g m_g, D = sum_g m_g^2, S0 = sum_g deg_g, G = number of bins.
    """
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    for i in range(values.shape[0]):
        sums[bin_of_cell[i]] += values[i]
        counts[bin_of_cell[i]] += 1.0
    means = sums / counts
    lags = np.zeros(n_bins, dtype=np.float64)
    stats = np.zeros(6, dtype=np.float64)
    for g in range(n_bins):
        deg = b_indptr[g + 1] - b_indptr[g]
        for p in range(b_indptr[g], b_indptr[g + 1]):
            lags[g] += means[b_indices[p]]
        stats[0] += means[g] * lags[g]
        stats[1] += means[g] * deg
        stats[2] += means[g]
        stats[3] += means[g] * means[g]
        stats[4] += deg
    stats[5] = n_bins
    return means, counts, lags, stats


@njit(cache=True)
def _grid_morans(a_, b_, c_, d_, s0, g):
    if s0 <= 0 or g <= 1:
        return 0.0
    mean = c_ / g
    numer = a_ - 2.0 * mean * b_ + mean * mean * s0
    denom = d_ - c_ * c_ / g
    if denom <= 0:
        return 0.0
    return (g / s0) * numer / denom


@njit(cache=True)
def _bin_adjacent(g, h, b_indptr, b_indices):
    for p in range(b_indptr[g], b_indptr[g + 1]):
        if b_indices[p] == h:
            return 1.0
    return 0.0


@njit(cache=True)
def _values_greedy(
    values,
    indptr,
    indices,
    targets,
    bins,
    b_indptr,
    b_indices,
    b_offsets,
    eps,
    max_proposals,
    stall,
):
    """Swap values until Moran's I at every scale is within eps of its target.

    targets[0] is the cell-graph Moran's I; targets[1 + s] is that of coarse
    scale s, whose bins are numbered b_offsets[s]..b_offsets[s+1] in the
    concatenated bin graph and bins[s, i] gives cell i's bin. Returns the
    achieved Moran's I per scale.
    """
    n = values.shape[0]
    n_scales = bins.shape[0]
    lags, numer, scale = _values_moran_state(values, indptr, indices)
    total_bins = b_offsets[n_scales]
    means = np.zeros(total_bins, dtype=np.float64)
    counts = np.zeros(total_bins, dtype=np.float64)
    blags = np.zeros(total_bins, dtype=np.float64)
    stats = np.zeros((n_scales, 6), dtype=np.float64)
    morans = np.zeros(n_scales + 1, dtype=np.float64)
    morans[0] = numer * scale
    for s in range(n_scales):
        lo = b_offsets[s]
        hi = b_offsets[s + 1]
        local_indptr = b_indptr[lo : hi + 1] - b_indptr[lo]
        local_indices = b_indices[b_indptr[lo] : b_indptr[hi]] - lo
        m, c, lg, st = _grid_state(
            values, bins[s], local_indptr, local_indices, hi - lo
        )
        means[lo:hi] = m
        counts[lo:hi] = c
        blags[lo:hi] = lg
        stats[s, :] = st
        morans[s + 1] = _grid_morans(st[0], st[1], st[2], st[3], st[4], st[5])

    new_stats = np.zeros((n_scales, 4), dtype=np.float64)
    new_morans = np.zeros(n_scales + 1, dtype=np.float64)
    rejections = 0
    for proposal in range(max_proposals):
        if proposal % 32 == 0:
            worst = 0.0
            for s in range(n_scales + 1):
                dev = abs(targets[s] - morans[s])
                if dev > worst:
                    worst = dev
            if worst < eps:
                break
        a = np.random.randint(n)  # noqa: NPY002
        b = np.random.randint(n)  # noqa: NPY002
        delta = values[b] - values[a]
        if delta == 0.0:
            continue
        w_ab = _is_neighbour(a, b, indptr, indices)
        d_numer = 2.0 * delta * (lags[a] - lags[b]) - 2.0 * w_ab * delta * delta
        new_morans[0] = (numer + d_numer) * scale
        for s in range(n_scales):
            ga = b_offsets[s] + bins[s, a]
            gb = b_offsets[s] + bins[s, b]
            if ga == gb:
                new_morans[s + 1] = morans[s + 1]
                for q in range(4):
                    new_stats[s, q] = stats[s, q]
                continue
            # Cell a takes value b's value, so bin ga's mean moves by
            # +delta/count and bin gb's by -delta/count.
            d_a = delta / counts[ga]
            d_b = -delta / counts[gb]
            w_g = _bin_adjacent(ga, gb, b_indptr, b_indices)
            deg_a = b_indptr[ga + 1] - b_indptr[ga]
            deg_b = b_indptr[gb + 1] - b_indptr[gb]
            new_stats[s, 0] = stats[s, 0] + 2.0 * (
                d_a * blags[ga] + d_b * blags[gb] + w_g * d_a * d_b
            )
            new_stats[s, 1] = stats[s, 1] + d_a * deg_a + d_b * deg_b
            new_stats[s, 2] = stats[s, 2] + d_a + d_b
            new_stats[s, 3] = (
                stats[s, 3]
                + 2.0 * means[ga] * d_a
                + d_a * d_a
                + 2.0 * means[gb] * d_b
                + d_b * d_b
            )
            new_morans[s + 1] = _grid_morans(
                new_stats[s, 0],
                new_stats[s, 1],
                new_stats[s, 2],
                new_stats[s, 3],
                stats[s, 4],
                stats[s, 5],
            )
        d_energy = 0.0
        for s in range(n_scales + 1):
            d_energy += (targets[s] - new_morans[s]) ** 2 - (
                targets[s] - morans[s]
            ) ** 2
        if d_energy <= 0.0:
            values[a], values[b] = values[b], values[a]
            for p in range(indptr[a], indptr[a + 1]):
                lags[indices[p]] += delta
            for p in range(indptr[b], indptr[b + 1]):
                lags[indices[p]] -= delta
            numer += d_numer
            morans[0] = new_morans[0]
            for s in range(n_scales):
                ga = b_offsets[s] + bins[s, a]
                gb = b_offsets[s] + bins[s, b]
                morans[s + 1] = new_morans[s + 1]
                if ga == gb:
                    continue
                d_a = delta / counts[ga]
                d_b = -delta / counts[gb]
                means[ga] += d_a
                means[gb] += d_b
                for p in range(b_indptr[ga], b_indptr[ga + 1]):
                    blags[b_indices[p]] += d_a
                for p in range(b_indptr[gb], b_indptr[gb + 1]):
                    blags[b_indices[p]] += d_b
                for q in range(4):
                    stats[s, q] = new_stats[s, q]
            rejections = 0
        else:
            rejections += 1
            if rejections > stall:
                break
    return morans


@njit(cache=True)
def _all_scale_morans(values, indptr, indices, bins, b_indptr, b_indices, b_offsets):
    n_scales = bins.shape[0]
    out = np.zeros(n_scales + 1, dtype=np.float64)
    _, numer, scale = _values_moran_state(values, indptr, indices)
    out[0] = numer * scale
    for s in range(n_scales):
        lo = b_offsets[s]
        hi = b_offsets[s + 1]
        local_indptr = b_indptr[lo : hi + 1] - b_indptr[lo]
        local_indices = b_indices[b_indptr[lo] : b_indptr[hi]] - lo
        _, _, _, st = _grid_state(values, bins[s], local_indptr, local_indices, hi - lo)
        out[s + 1] = _grid_morans(st[0], st[1], st[2], st[3], st[4], st[5])
    return out


def _gaussian_field(
    coords: npt.NDArray[np.float64],
    length: float,
    pixel: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Smooth Gaussian random field with the given length scale, sampled at
    the cell coordinates (FFT filtering of white noise on a pixel grid)."""
    origin = coords.min(axis=0) - 3 * length
    shape = np.ceil((coords.max(axis=0) + 3 * length - origin) / pixel).astype(int) + 1
    shape = np.minimum(shape, 4096)
    pixel_eff = (coords.max(axis=0) + 3 * length - origin) / (shape - 1)
    noise = rng.standard_normal(tuple(shape))
    fx = np.fft.fftfreq(shape[0], d=pixel_eff[0])
    fy = np.fft.rfftfreq(shape[1], d=pixel_eff[1])
    kernel = np.exp(-2 * (np.pi * length) ** 2 * (fx[:, None] ** 2 + fy[None, :] ** 2))
    field = np.fft.irfft2(np.fft.rfft2(noise) * kernel, s=tuple(shape))
    ij = np.round((coords - origin) / pixel_eff).astype(int)
    return field[ij[:, 0], ij[:, 1]]


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)  # noqa: NPY002


@njit(parallel=True, cache=True)
def _values_greedy_many(
    starts,
    indptr,
    indices,
    targets,
    bins,
    b_indptr,
    b_indices,
    b_offsets,
    eps,
    max_proposals,
    stall,
    seed,
):
    n_perms = starts.shape[0]
    achieved = np.zeros((n_perms, targets.shape[0]), dtype=np.float64)
    for p in prange(n_perms):
        np.random.seed(seed + p)  # noqa: NPY002
        values = starts[p]
        achieved[p, :] = _values_greedy(
            values,
            indptr,
            indices,
            targets,
            bins,
            b_indptr,
            b_indices,
            b_offsets,
            eps,
            max_proposals,
            stall,
        )
    return achieved


def _multiscale_structure(
    coords: npt.NDArray[np.float64] | None,
    radius: float | None,
    coarse_scales: tuple[float, ...],
    n: int,
):
    if coords is None or radius is None or len(coarse_scales) == 0:
        return (
            np.zeros((0, n), dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        )
    bins_list, adjs, offsets = [], [], [0]
    for mult in coarse_scales:
        b_of_c, adj = grid_structure(coords, radius * mult)
        if adj.shape[0] < MIN_BINS:
            continue  # Too few bins for a meaningful Moran's I.
        bins_list.append(b_of_c)
        adjs.append(adj)
        offsets.append(offsets[-1] + adj.shape[0])
    if not adjs:
        return _multiscale_structure(None, None, (), n)
    block = sp.csr_matrix(sp.block_diag(adjs, format="csr"))
    block.sort_indices()
    return (
        np.vstack(bins_list),
        block.indptr.astype(np.int64),
        block.indices.astype(np.int64),
        np.array(offsets, dtype=np.int64),
    )


def values_morans_i(
    values: npt.NDArray[np.floating],
    graph: sp.csr_matrix,
    coords: npt.NDArray[np.float64] | None = None,
    radius: float | None = None,
    coarse_scales: tuple[float, ...] = (4.0, 16.0),
) -> npt.NDArray[np.float64]:
    """Moran's I of a continuous variable on the cell graph, then (if coords
    and radius are given) of its bin means at each coarse scale."""
    graph = sp.csr_matrix(graph)
    values = np.asarray(values, dtype=np.float64)
    structure = _multiscale_structure(coords, radius, coarse_scales, len(values))
    return _all_scale_morans(
        values,
        graph.indptr.astype(np.int64),
        graph.indices.astype(np.int64),
        *structure,
    )


def permute_values(
    values: npt.NDArray[np.floating],
    graph: sp.csr_matrix,
    n_perms: int,
    coords: npt.NDArray[np.float64] | None = None,
    radius: float | None = None,
    coarse_scales: tuple[float, ...] = (4.0, 16.0),
    eps: float = 0.005,
    max_proposals_per_cell: int = 200,
    stall_per_cell: int = 10,
    random_state: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Surrogates of a gene's expression keeping its values and spatial
    autocorrelation.

    Each surrogate is a permutation of values across cells, so the expression
    distribution is exact. Moran's I on the cell graph, and, when coords and
    radius are given, Moran's I of bin means on square grids with bins of
    radius * coarse_scales, are matched to within eps (Arthur 2025). Starting
    points are smooth Gaussian random fields whose length scale is increased
    until every scale is at or above its target, so targets are approached
    from above. This is the autocorrelation-aware null for LR co-localisation
    motivated by SOAAR (Khatri et al., bioRxiv 2026): genes are spatially
    structured, so testing LR co-localisation against spatially random genes
    gives too many false positives.

    Returns
    -------
    surrogates: (n_perms, n_cells)
    achieved: (n_perms, n_scales) Moran's I of each surrogate per scale
    observed: (n_scales,) Moran's I of values per scale (cell graph first)
    """
    graph = sp.csr_matrix(graph)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    indptr = graph.indptr.astype(np.int64)
    indices = graph.indices.astype(np.int64)
    structure = _multiscale_structure(coords, radius, coarse_scales, n)
    observed = _all_scale_morans(values, indptr, indices, *structure)

    sorted_vals = np.sort(values)
    starts = np.zeros((n_perms, n), dtype=np.float64)
    last_length = None
    for p in range(n_perms):
        rng = np.random.default_rng(random_state + p)
        if coords is None or radius is None or not np.any(observed > 0):
            starts[p] = rng.permutation(values)
            continue
        extent = float(np.ptp(coords, axis=0).max())
        # Start just below the length scale that worked for the last surrogate.
        length = (
            radius / 2 if last_length is None else max(radius / 2, last_length / 1.5)
        )
        while True:
            field = _gaussian_field(coords, length, max(radius / 2, length / 4), rng)
            starts[p, np.argsort(field)] = sorted_vals
            init = _all_scale_morans(starts[p], indptr, indices, *structure)
            if np.all(init >= observed) or length > extent:
                break
            length *= 1.5
        last_length = length

    achieved = _values_greedy_many(
        starts,
        indptr,
        indices,
        observed,
        *structure,
        float(eps),
        int(max_proposals_per_cell * n),
        int(stall_per_cell * n),
        int(random_state),
    )
    return starts, achieved, observed
