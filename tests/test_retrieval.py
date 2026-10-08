"""
test_retrieval.py
-----------------
Tests that hybrid BM25 + FAISS retrieval returns sensible results for
maintenance-domain queries against the Haas knowledge base.

These tests require the pre-built knowledge-base artefacts in
outputs/knowledge_base/.  They are skipped automatically if those
files are not present.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

KB_DIR = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "knowledge_base"
)

REQUIRED_FILES = [
    KB_DIR / "maintenance_chunks.csv",
    KB_DIR / "maintenance_faiss.index",
]


# Skip the entire module if the knowledge base has not been built
for _f in REQUIRED_FILES:
    if not _f.exists():
        pytest.skip(
            f"Knowledge base file not found: {_f}. "
            "Run knowledge_base_builder.py first.",
            allow_module_level=True,
        )


from hybrid_retrieval import (
    load_chunks,
    load_faiss,
    build_bm25,
    load_embedding_model,
    search,
)


# ------------------------------------------------------------------ #
# Module-level setup — load components once for the whole module      #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def retrieval_components():
    chunks_df       = load_chunks()
    faiss_index     = load_faiss()
    bm25            = build_bm25(chunks_df)
    embedding_model = load_embedding_model()
    return chunks_df, bm25, embedding_model, faiss_index


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def do_search(components, query):
    chunks_df, bm25, model, faiss_index = components
    return search(query, chunks_df, bm25, model, faiss_index)


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

MAINTENANCE_QUERIES = [
    "what causes excessive spindle vibration",
    "spindle bearing noise and overheating",
    "what should I inspect when spindle vibration increases",
    "spindle lubrication problem",
    "toolholder causing chatter",
]


class TestRetrievalReturnsResults:

    @pytest.mark.parametrize("query", MAINTENANCE_QUERIES)
    def test_returns_at_least_one_result(self, retrieval_components, query):
        results = do_search(retrieval_components, query)
        assert len(results) >= 1, f"No results for query: {query!r}"

    @pytest.mark.parametrize("query", MAINTENANCE_QUERIES)
    def test_result_has_required_fields(self, retrieval_components, query):
        results = do_search(retrieval_components, query)
        required_keys = {"source", "page", "text", "hybrid_score"}
        for r in results:
            assert required_keys.issubset(r.keys()), (
                f"Result missing keys. Got: {list(r.keys())}"
            )

    @pytest.mark.parametrize("query", MAINTENANCE_QUERIES)
    def test_hybrid_score_is_positive(self, retrieval_components, query):
        results = do_search(retrieval_components, query)
        for r in results:
            assert r["hybrid_score"] >= 0, (
                f"Negative hybrid score: {r['hybrid_score']}"
            )

    @pytest.mark.parametrize("query", MAINTENANCE_QUERIES)
    def test_results_sorted_by_score_descending(self, retrieval_components, query):
        results = do_search(retrieval_components, query)
        scores = [r["hybrid_score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Results are not sorted by hybrid_score descending"
        )

    def test_spindle_vibration_returns_haas_content(self, retrieval_components):
        """The top result for a spindle-vibration query should come from the Haas manuals."""
        results = do_search(retrieval_components, "excessive spindle vibration")
        sources = [r["source"].lower() for r in results]
        assert any("haas" in s or "vf" in s or "spindle" in s or "operator" in s
                   for s in sources), (
            f"Expected Haas-related source, got: {sources}"
        )

    def test_page_number_is_integer(self, retrieval_components):
        results = do_search(retrieval_components, "spindle lubrication")
        for r in results:
            assert isinstance(int(r["page"]), int)

    def test_text_is_non_empty_string(self, retrieval_components):
        results = do_search(retrieval_components, "toolholder chatter")
        for r in results:
            assert isinstance(r["text"], str) and len(r["text"].strip()) > 0
