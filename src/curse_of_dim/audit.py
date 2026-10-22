"""Offline geometric audit of user-supplied document/query embedding arrays.

NPZ keys: doc_embeddings, query_embeddings, and optionally doc_labels,
query_labels. This command never downloads models or uploads vector data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .advanced import effective_rank, embedding_transformations, retrieval_metrics


def audit_npz(input_file: str | Path, output: str | Path, k: int = 5) -> dict:
    """Evaluate cosine-retrieval geometry using train-on-doc transforms only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(input_file, allow_pickle=False) as archive:
        needed = {"doc_embeddings", "query_embeddings"}
        if not needed.issubset(archive.files):
            raise ValueError(f"NPZ missing required keys {sorted(needed - set(archive.files))}")
        docs = np.asarray(archive["doc_embeddings"], dtype="float64")
        queries = np.asarray(archive["query_embeddings"], dtype="float64")
        has_doc = "doc_labels" in archive.files
        has_query = "query_labels" in archive.files
        if has_doc != has_query:
            raise ValueError("Provide both doc_labels and query_labels, or neither")
        dl = archive["doc_labels"] if has_doc else None
        ql = archive["query_labels"] if has_query else None
        # Refuse object/pickled labels; keep this a local, data-only audit.
        if dl is not None and (dl.dtype.kind not in "iufUS" or ql.dtype.kind not in "iufUS"):
            raise ValueError("Labels must be numeric or unicode strings (no object arrays)")
    if (docs.ndim != 2 or queries.ndim != 2 or docs.shape[1] != queries.shape[1]
            or len(docs) < 3 or len(queries) < 1 or not np.isfinite(docs).all()
            or not np.isfinite(queries).all()):
        raise ValueError("Require finite, nonempty, matching 2D embedding matrices")
    if k > len(docs) or k <= 0:
        raise ValueError("k must be within database size")
    spectral = effective_rank(docs)
    transforms = embedding_transformations(docs, queries, whiten_components=min(16, docs.shape[1]))
    evaluations = {name: retrieval_metrics(d, q, dl, ql, k=k)
                   for name, (d,q) in transforms.items()}
    report = {"samples":{"documents":len(docs),"queries":len(queries),"dimension":docs.shape[1]},
              "k":k,"spectral":spectral,"transforms":evaluations,
              "notes":["All transforms fit using documents only; queries remain held out.",
                       "Without labels precision/hit are not estimated; hubness does not imply semantic quality.",
                       "Cosine geometry changes with centering and component removal.",
                       "PCA entropy rank is NOT a manifold dimension estimator."]}
    (output / "audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                                       encoding="utf-8")
    names = list(evaluations)
    fig,ax=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
    ax[0].bar(names,[evaluations[n]["hubness_gini"] for n in names])
    ax[0].set(title=f"Cosine kNN hubness (k={k})",ylabel="k-occurrence Gini",ylim=(0,1))
    metric="precision_at_k" if dl is not None else "max_neighbor_occurrences"
    ax[1].bar(names,[evaluations[n][metric] for n in names])
    ax[1].set(title=metric, ylabel=metric)
    for a in ax:
        a.tick_params(axis="x",rotation=20)
    fig.savefig(output / "audit.png",dpi=145,bbox_inches="tight")
    plt.close(fig)
    return report


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog="curse-of-dim-audit",description="Offline, query-held-out embedding diagnostics")
    parser.add_argument("--input",type=Path,required=True,help="Local NPZ with document/query embeddings")
    parser.add_argument("--out",type=Path,default=Path("outputs/audit"))
    parser.add_argument("--k",type=int,default=5)
    args=parser.parse_args(argv)
    report=audit_npz(args.input,args.out,args.k)
    print(json.dumps({"saved":str(args.out.resolve()),"samples":report["samples"]},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
