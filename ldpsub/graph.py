"""Loading SNAP graphs."""

import gzip

import networkx as nx


def load_edge_list(path):
    """Load a SNAP edge list as a simple undirected graph.

    - Lines starting with '#' are comments.
    - Edge direction is ignored, duplicate edges and self-loops are removed.
    - Nodes are relabeled 0..n-1.
    Works for plain text and .gz files.
    """
    opener = gzip.open if path.endswith(".gz") else open
    G = nx.Graph()
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            u, v = line.split()[:2]
            if u != v:
                G.add_edge(u, v)
    return nx.convert_node_labels_to_integers(G)
