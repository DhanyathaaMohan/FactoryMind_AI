"""
retrieval_service.py
--------------------
Service wrapper around hybrid_retrieval.  Loads retrieval components
once at first use and exposes a single search() function.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from hybrid_retrieval import (
    load_chunks,
    load_faiss,
    build_bm25,
    load_embedding_model,
    search as hybrid_search,
)


# --------------------------------------------------
# Lazy module-level cache
# --------------------------------------------------

_loaded = False
_chunks_df = None
_faiss_index = None
_bm25 = None
_embedding_model = None


def _ensure_loaded():
    global _loaded, _chunks_df, _faiss_index, _bm25, _embedding_model

    if not _loaded:
        print("Loading retrieval components …")
        _chunks_df     = load_chunks()
        _faiss_index   = load_faiss()
        _bm25          = build_bm25(_chunks_df)
        _embedding_model = load_embedding_model()
        _loaded = True
        print("Retrieval components ready.")


def is_loaded() -> bool:
    return _loaded


def search(query: str, top_k: int = 3) -> list:
    """
    Hybrid BM25 + FAISS search over the Haas knowledge base.

    Parameters
    ----------
    query : str
    top_k : int   number of results to return (default 3)

    Returns
    -------
    list[dict]  — each dict has keys: source, page, hybrid_score, text
    """
    _ensure_loaded()

    results = hybrid_search(
        query,
        _chunks_df,
        _bm25,
        _embedding_model,
        _faiss_index,
    )

    # Return at most top_k results, with a preview field
    out = []
    for r in results[:top_k]:
        out.append({
            "source":       r["source"],
            "page":         int(r["page"]),
            "hybrid_score": round(float(r["hybrid_score"]), 4),
            "preview":      str(r["text"])[:300],
        })
    return out
