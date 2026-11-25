"""Per-node sums over a node's neighbor block, shared by the exact counts and the LDP estimators.

B is the block of a symmetric matrix among node i's neighbors:
true 0/1 adjacency on the exact side, centered noisy bits y - q on the LDP side.
The diagonal of B is always ignored (it is not a pair).
"""

import numpy as np
from scipy import sparse


def neighbor_block(M, N):
    """M restricted to rows and columns N, as a dense float array (M dense or sparse)."""
    block = M[N][:, N]
    if sparse.issparse(block):
        block = block.toarray()
    return block.astype(float)


def _zero_diagonal(B):
    B = B.copy()
    np.fill_diagonal(B, 0)
    return B


def pair_sum(B):
    """Sum over pairs {k, l}: sum_{k<l} B_kl."""
    return np.triu(B, k=1).sum()


def wedge_sum(B):
    """Sum over wedges centered at j: sum_j sum_{{k,l}, k,l != j} B_jk B_jl.

    For row j with entries b_k: sum_{k<l} b_k b_l = ((sum_k b_k)^2 - sum_k b_k^2) / 2.
    """
    B = _zero_diagonal(B)
    row_sums = B.sum(axis=1)
    row_squares = (B**2).sum(axis=1)
    return ((row_sums**2 - row_squares) / 2).sum()


def triangle_sum(B):
    """Sum over triangles {j, k, l}: sum_{j<k<l} B_jk B_jl B_kl  (B symmetric).

    (B @ B)_jl = sum_k B_jk B_kl, so sum_{j,l} (B @ B)_jl B_jl counts each triangle 6 times.
    """
    B = _zero_diagonal(B)
    with np.errstate(all="ignore"):  # Apple Accelerate BLAS gives false warnings in matmul
        BB = B @ B
    return (BB * B).sum() / 6
