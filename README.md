# ldp-subgraph-counting

Counting the six connected 4-node subgraphs (3-star, 3-path, tailed triangle, 4-cycle, diamond, 4-clique)
under edge local differential privacy (edge-LDP), with a two-round protocol in the style of Imola et al. (2021).

CMPT 982 project, SFU.

## Layout

```
ldpsub/    library code (loading, ground truth, privacy tools, estimators)
tests/     unit tests and sanity checks (pytest)
scripts/   experiment and plotting scripts
data/      SNAP datasets (downloaded, not committed)
results/   experiment outputs and plots (not committed)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets (SNAP)

- `ca-GrQc` (debugging)
- `ego-Facebook`, `ca-AstroPh` (main experiments)
- `email-Enron` (optional)

Download all of them into `data/` (run from the repo root):

```bash
python -m scripts.download_data
```

In code: `load_dataset(name, n=None, seed=None)` from `ldpsub.datasets` (`n` samples a random induced subgraph).

## Tests

```bash
pytest
```

## References

- J. Imola, T. Murakami, K. Chaudhuri. Locally Differentially Private Analysis of Graph Statistics. USENIX Security 2021.
- A. Pinar, C. Seshadhri, V. Vishal. ESCAPE: Efficiently Counting All 5-Vertex Subgraphs. WWW 2017.
- Guo et al. Quadrilateral (4-cycle) counting under edge-LDP (QuaTR). arXiv:2507.06508, 2025.
- J. Leskovec, A. Krevl. SNAP Datasets: Stanford Large Network Dataset Collection. http://snap.stanford.edu/data, 2014.

## License

MIT, see [LICENSE](LICENSE).
