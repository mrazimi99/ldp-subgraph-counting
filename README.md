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

Download the edge lists from https://snap.stanford.edu/data/ into `data/`.

## Tests

```bash
pytest
```

## License

MIT, see [LICENSE](LICENSE).
