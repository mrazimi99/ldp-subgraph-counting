"""Edge-LDP estimators for the six 4-node patterns, two-round protocol of Imola et al. (USENIX Security 2021).

Node i only uses its own neighbor list N_i and the public round-1 data.
Running a local value on noiseless public data (true degrees, Y = A, p = 1, q = 0)
gives the exact local value, so each estimator can be checked against the exact count.
"""

from dataclasses import dataclass
from functools import partial
from math import comb
from typing import Optional

import networkx as nx
import numpy as np

from ldpsub.local import neighbor_block, pair_sum, triangle_sum, wedge_sum
from ldpsub.privacy import (
    laplace,
    max_degree_bound,
    noisy_degrees,
    randomized_response,
    rr_probabilities,
)


@dataclass
class Public:
    """What the server publishes after round 1."""

    d_noisy: np.ndarray       # noisy degrees of all nodes
    D: int                    # degree bound
    Y: Optional[np.ndarray]   # noisy graph, None if the pattern does not use RR
    p: float                  # RR keep probability
    q: float                  # RR flip probability


def degrees(G):
    return np.array([G.degree(i) for i in range(G.number_of_nodes())])


def run_round1(G, eps0, eps1, rng):
    """Nodes send noisy degrees (and RR bits if eps1 is given); the server publishes D and Y."""
    d_noisy = noisy_degrees(degrees(G), eps0, rng)
    D = max_degree_bound(d_noisy)
    if eps1 is None:
        return Public(d_noisy, D, None, 1.0, 0.0)
    p, q = rr_probabilities(eps1)
    return Public(d_noisy, D, randomized_response(G, eps1, rng), p, q)


def noiseless_public(G):
    """Round-1 data without noise: true degrees, D = max degree, Y = A, p = 1, q = 0."""
    d = degrees(G)
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()), dtype=np.uint8)
    return Public(d.astype(float), int(d.max()), A, 1.0, 0.0)


def run_round2(local_value, neighbors, public, sensitivity, eps2, rng):
    """Each node i sends local_value(i, N_i, public) + Lap(sensitivity / eps2); the server sums."""
    values = np.array([local_value(i, N_i, public) for i, N_i in enumerate(neighbors)], dtype=float)  # nodes
    noisy_values = values + laplace(sensitivity / eps2, len(values), rng)                             # nodes
    return noisy_values.sum()                                                                         # server


# 3-star: local value C(d_i, 3), sensitivity C(D, 2). Uses only eps0 (for D) and eps2.

def local_3star(i, N_i, public):
    return comb(len(N_i), 3)


def round2_3star(public, neighbors, eps2, rng):
    return run_round2(local_3star, neighbors, public, comb(public.D, 2), eps2, rng)


def triangle_building_block(N_i, public):
    """w_i = sum over pairs {k, l} in N_i of centered noisy bits y_kl - q.  E[w_i] = (p - q) t_i."""
    return pair_sum(neighbor_block(public.Y, N_i) - public.q)


# Tailed triangle: X_i = w_i (d_i - 2). Sensitivity (3/2) D (D - 1). Server divides the sum by (p - q).

def local_tailed_triangle(i, N_i, public):
    w_i = triangle_building_block(N_i, public)
    return w_i * (len(N_i) - 2)


def round2_tailed_triangle(public, neighbors, eps2, rng):
    D = public.D
    total = run_round2(local_tailed_triangle, neighbors, public, 1.5 * D * (D - 1), eps2, rng)
    return total / (public.p - public.q)


# 3-path: U_i = S_i / 2 - w_i / (p - q), S_i = (d_i - 1) * sum_{j in N_i} max(d~_j - 1, 0).
# Summed over nodes: sum_edges (d_i - 1)(d_j - 1) - 3T. Sensitivity D^2 + D / (p - q). Server sums.

def local_3path(i, N_i, public):
    d_i = len(N_i)
    S_i = (d_i - 1) * np.maximum(public.d_noisy[N_i] - 1, 0).sum()
    w_i = triangle_building_block(N_i, public)
    return S_i / 2 - w_i / (public.p - public.q)


def round2_3path(public, neighbors, eps2, rng):
    D = public.D
    sensitivity = D**2 + D / (public.p - public.q)
    return run_round2(local_3path, neighbors, public, sensitivity, eps2, rng)


# 4-clique: Z_i = sum over triples {j, k, l} in N_i of centered noisy bits y_jk y_jl y_kl,
# E[Z_i] = (p - q)^3 * (triangles among N_i). Sensitivity C(D, 2). Server divides the sum by 4 (p - q)^3.

def local_4clique(i, N_i, public):
    return triangle_sum(neighbor_block(public.Y, N_i) - public.q)


def round2_4clique(public, neighbors, eps2, rng):
    total = run_round2(local_4clique, neighbors, public, comb(public.D, 2), eps2, rng)
    return total / (4 * (public.p - public.q) ** 3)


# Diamond: Y_i = sum over j in N_i, pairs {k, l} in N_i \ {j} of centered noisy bits y_jk y_jl,
# E[Y_i] = (p - q)^2 * (wedges among N_i). Each diamond is seen from both ends of its middle edge.
# Sensitivity (3/2) D (D - 1). Server divides the sum by 2 (p - q)^2.

def local_diamond(i, N_i, public):
    return wedge_sum(neighbor_block(public.Y, N_i) - public.q)


def round2_diamond(public, neighbors, eps2, rng):
    D = public.D
    total = run_round2(local_diamond, neighbors, public, 1.5 * D * (D - 1), eps2, rng)
    return total / (2 * (public.p - public.q) ** 2)


# 4-cycle (method B, close to QuaTR of Guo et al., arXiv 2507.06508):
# for each pair {k, l} in N_i, c^_kl = sum_{m != i, k, l} a^_km a^_lm estimates the common neighbors
# of k and l other than i, with debiased bits a^ = (y - q) / (p - q).
# s_i = sum over pairs of clip_tau(c^_kl). Sensitivity D * tau. Server divides the sum by 4.

def debiased_rows(i, N_i, public):
    """Debiased noisy bits a^_km for k in N_i and all nodes m (d x n, float32).

    Zero at column i (m != i) and at each row's own node (m != k), so these terms drop out of R @ R.T.
    """
    q = np.float32(public.q)               # float32 scalars, so numpy does not promote R to float64
    p_minus_q = np.float32(public.p - public.q)
    R = (public.Y[N_i].astype(np.float32) - q) / p_minus_q
    R[:, i] = 0
    for row, k in enumerate(N_i):
        R[row, k] = 0
    return R


def local_4cycle(i, N_i, public, tau):
    R = debiased_rows(i, N_i, public)
    with np.errstate(all="ignore"):  # Apple Accelerate BLAS gives false warnings in matmul
        C_hat = R @ R.T              # C_hat[k, l] = c^_kl
    return pair_sum(np.clip(C_hat, -tau, tau))


def round2_4cycle(public, neighbors, eps2, rng, tau):
    total = run_round2(partial(local_4cycle, tau=tau), neighbors, public, public.D * tau, eps2, rng)
    return total / 4
