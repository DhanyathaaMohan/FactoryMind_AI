"""
rag_service.py
--------------
Service wrapper around rag_assistant.  Loads the generation model
once and exposes ask() for grounded maintenance answer generation.
"""

import sys
from pathlib import Path

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_assistant import load_generation_model, answer_query

# Import the retrieval service using its package path to avoid bare-name
# resolution issues when the backend is imported from the project root.
from backend.services import retrieval_service as _rs


# --------------------------------------------------
# Lazy cache
# --------------------------------------------------

_gen_loaded = False
_tokenizer  = None
_gen_model  = None


def _ensure_generation():
    global _gen_loaded, _tokenizer, _gen_model

    if not _gen_loaded:
        print("Loading generation model …")
        _tokenizer, _gen_model = load_generation_model()
        _gen_loaded = True
        print("Generation model ready.")


def is_generation_loaded() -> bool:
    return _gen_loaded


# --------------------------------------------------
# Public API
# --------------------------------------------------

def ask(query: str, predicted_rul: float = None) -> tuple:
    """
    Generate a grounded maintenance answer.

    Parameters
    ----------
    query         : str    maintenance question
    predicted_rul : float  normalised RUL (optional, for context)

    Returns
    -------
    (answer: str, sources: list[dict])
    """
    _rs._ensure_loaded()
    _ensure_generation()

    answer, retrieved = answer_query(
        query=query,
        predicted_rul=predicted_rul,
        chunks_df=_rs._chunks_df,
        bm25=_rs._bm25,
        embedding_model=_rs._embedding_model,
        faiss_index=_rs._faiss_index,
        tokenizer=_tokenizer,
        generation_model=_gen_model,
    )

    sources = []
    for r in retrieved[:3]:
        sources.append({
            "source":       r["source"],
            "page":         int(r["page"]),
            "hybrid_score": round(float(r["hybrid_score"]), 4),
            "preview":      str(r["text"])[:300],
        })

    return answer, sources
