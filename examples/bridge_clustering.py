"""Bridge example: lexindex (string ids) <-> betula-cluster (clusters).

The two libraries compose cleanly. ``betula-cluster`` clusters *numeric* rows and returns an
integer label per row; ``lexindex`` owns the mapping between your *string* ids and the dense
``[0, n)`` integer rows. Use the lexindex dense id as the canonical row index of the embedding
matrix, cluster it, and you can answer both directions:

  - which cluster is a given string id in?            string id -> row -> label
  - which string ids are in a given cluster?          label -> rows -> string ids

Run::

    pip install lexindex betula-cluster numpy
    python examples/bridge_clustering.py
"""

from __future__ import annotations

import betula_cluster
import numpy as np
from lexindex import PerfectHashIndex


def main() -> None:
    rng = np.random.default_rng(0)
    n_per, dim, k = 300, 16, 4

    # Synthetic corpus: each document has a STRING id and an embedding drawn near one of k centers.
    ids = [f"doc-{i:05d}" for i in range(n_per * k)]
    true_cluster = np.repeat(np.arange(k), n_per)
    centers = rng.normal(scale=6.0, size=(k, dim))
    emb = (centers[true_cluster] + rng.normal(size=(len(ids), dim))).astype(np.float64)

    # lexindex: the authority for string id <-> dense [0, n) id (here, also a fixed vocabulary).
    idx = PerfectHashIndex(ids)

    # Place each embedding at its dense-id row, so the row index *is* the lexindex id. Track the
    # true cluster in the same (dense-id) order so it stays aligned with the labels below.
    matrix = np.empty_like(emb)
    true_by_id = np.empty(len(ids), dtype=np.int64)
    for src_row, doc_id in enumerate(ids):
        slot = idx.id(doc_id)
        matrix[slot] = emb[src_row]
        true_by_id[slot] = true_cluster[src_row]

    # betula-cluster: one integer label per row (i.e. per dense id).
    labels = np.asarray(betula_cluster.fit_predict(matrix, n_clusters=k, method="kmeans", seed=0))

    def cluster_of(doc_id: str) -> int:
        """string id -> cluster (via the lexindex dense id)."""
        row = idx.id(doc_id)
        if row is None:
            raise KeyError(doc_id)
        return int(labels[row])

    def members(cluster: int) -> list[str]:
        """cluster -> string ids (reverse lexindex lookup on the matching rows)."""
        return [idx.key(int(r)) for r in np.flatnonzero(labels == cluster)]

    print(f"{len(ids)} documents, {k} clusters\n")

    print("string id -> cluster:")
    for doc_id in ["doc-00000", "doc-00450", "doc-00900", "doc-01150"]:
        print(f"  {doc_id} -> cluster {cluster_of(doc_id)}")

    c = cluster_of("doc-00000")
    m = members(c)
    print(f"\ncluster {c}: {len(m)} documents, e.g. {m[:5]}")

    # Sanity: the planted structure is recovered — each predicted cluster is dominated by one true
    # cluster (clustering is unsupervised, so label *numbers* differ; purity is what matters).
    purity = sum(np.bincount(true_by_id[labels == c], minlength=k).max() for c in range(k)) / len(
        ids
    )
    print(f"\nrecovered structure: purity = {purity:.3f}")
    assert purity > 0.95, "expected the bridge to recover the planted clusters"
    print("bridge OK")


if __name__ == "__main__":
    main()
