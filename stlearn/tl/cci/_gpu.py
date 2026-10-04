"""Optional torch implementations of the permutation counting steps.

The heavy, embarrassingly parallel parts of run_sc / run_cci_sc are counting
how many random gene pairs beat each cell's observed score, and counting cell
type pair edges under every surrogate labelling. Both map onto batched tensor
operations, so they run on a GPU when one is requested. The label and gene
resamplers (spatial_null.py) are sequential Markov chains of swaps and stay on
the CPU, parallel across surrogates with numba.
"""

import numpy as np
import numpy.typing as npt

# Upper bound on elements per temporary tensor, to keep GPU memory bounded.
_MAX_ELEMENTS = 50_000_000


def resolve_device(device: str | None):
    """Return a torch.device, or None to use the numba CPU path.

    device=None keeps the numba path. 'auto' picks CUDA, then Apple MPS, and
    otherwise falls back to numba. Any other string is passed to torch.device
    (e.g. 'cuda', 'cuda:1', 'mps', or 'cpu' for torch on the CPU).
    """
    if device is None:
        return None
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return None
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available.")
    return dev


def _float_dtype(dev):
    import torch

    # MPS has no float64.
    return torch.float32 if dev.type == "mps" else torch.float64


def count_bg_greater(
    obs_scores: npt.NDArray[np.float64],
    cell_idx: npt.NDArray[np.int64],
    lig_bg: npt.NDArray[np.float64],
    rec_bg: npt.NDArray[np.float64],
    pair_l: npt.NDArray[np.integer],
    pair_r: npt.NDArray[np.integer],
    min_expr: float,
    dev,
) -> npt.NDArray[np.int64]:
    """Torch version of single_cell._count_bg_greater."""
    import torch

    dtype = _float_dtype(dev)
    m = len(cell_idx)
    n_pairs = len(pair_l)
    out = np.zeros(m, dtype=np.int64)
    if m == 0:
        return out
    pl = torch.as_tensor(np.asarray(pair_l, dtype=np.int64), device=dev)
    pr = torch.as_tensor(np.asarray(pair_r, dtype=np.int64), device=dev)
    chunk = max(1, _MAX_ELEMENTS // max(n_pairs, 1))
    for start in range(0, m, chunk):
        cells = cell_idx[start : start + chunk]
        lig = torch.as_tensor(lig_bg[cells], dtype=dtype, device=dev)[:, pl]
        rec = torch.as_tensor(rec_bg[cells], dtype=dtype, device=dev)[:, pr]
        obs = torch.as_tensor(obs_scores[cells], dtype=dtype, device=dev)[:, None]
        hit = (lig > min_expr) & (rec > min_expr) & (lig * rec >= obs)
        out[start : start + len(cells)] = hit.sum(dim=1).cpu().numpy()
    return out


class EdgeCounter:
    """Keeps surrogate labellings on the device and counts, for a set of
    directed edges, how many surrogates give type pair counts >= observed."""

    def __init__(self, perms: npt.NDArray[np.integer], n_types: int, dev):
        import torch

        self.dev = dev
        self.n_types = n_types
        self.perms = torch.as_tensor(np.asarray(perms, dtype=np.int64), device=dev)

    def ge_counts(
        self,
        src: npt.NDArray[np.int64],
        dst: npt.NDArray[np.int64],
        observed: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        import torch

        k2 = self.n_types * self.n_types
        n_perms = self.perms.shape[0]
        src_t = torch.as_tensor(src, device=self.dev)
        dst_t = torch.as_tensor(dst, device=self.dev)
        obs_t = torch.as_tensor(observed.ravel(), device=self.dev)
        total = torch.zeros(k2, dtype=torch.int64, device=self.dev)
        batch = max(1, _MAX_ELEMENTS // max(len(src), 1))
        for start in range(0, n_perms, batch):
            labs = self.perms[start : start + batch]
            n_b = labs.shape[0]
            pair = labs[:, src_t] * self.n_types + labs[:, dst_t]
            offsets = torch.arange(n_b, device=self.dev)[:, None] * k2
            counts = torch.bincount(
                (pair + offsets).ravel(), minlength=n_b * k2
            ).reshape(n_b, k2)
            total += (counts >= obs_t[None, :]).sum(dim=0)
        return total.cpu().numpy().reshape(self.n_types, self.n_types)
