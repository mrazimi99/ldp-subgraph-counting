"""SNAP datasets: one list, one loading function."""

import os
import urllib.request

from ldpsub.graph import load_edge_list, sample_subgraph

DATASETS = {
    "ca-GrQc": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ego-Facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "ca-AstroPh": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
    "email-Enron": "https://snap.stanford.edu/data/email-Enron.txt.gz",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def dataset_path(name):
    return os.path.join(DATA_DIR, name + ".txt.gz")


def download(name):
    """Download a dataset into data/ if it is not there yet."""
    path = dataset_path(name)
    if not os.path.exists(path):
        print(f"{name}: downloading {DATASETS[name]}")
        urllib.request.urlretrieve(DATASETS[name], path)
    return path


def load_dataset(name, n=None, seed=None):
    """Load a dataset as a simple undirected graph with nodes 0..N-1.

    n=None: full graph. n=k: induced subgraph on k random nodes (see sample_subgraph).
    """
    G = load_edge_list(download(name))
    if n is not None:
        G = sample_subgraph(G, n, seed)
    return G
