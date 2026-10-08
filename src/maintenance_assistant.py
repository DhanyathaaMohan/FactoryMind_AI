"""
maintenance_assistant.py
------------------------
Top-level integration module that combines:

    1. predictive_maintenance  — RUL prediction + tool condition
    2. explainability          — SHAP feature influence
    3. hybrid_retrieval        — BM25 + FAISS knowledge-base search
    4. rag_assistant           — grounded maintenance answer generation

Main entry point
----------------
    analyze_and_answer(sensor_data, user_query) -> dict

Return schema
-------------
{
    "predicted_rul":    float,
    "tool_condition":   str,
    "top_shap_features": [
        {
            "feature":       str,
            "feature_value": float,
            "shap_value":    float,
            "direction":     str,
        },
        ...
    ],
    "retrieved_sources": [
        {
            "source":        str,
            "page":          int,
            "hybrid_score":  float,
            "preview":       str,
        },
        ...
    ],
    "maintenance_answer": str,
}
"""

import sys
from pathlib import Path

# Make src importable regardless of working directory
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from predictive_maintenance import predict
from explainability import explain_sample

from hybrid_retrieval import (
    load_chunks,
    load_faiss,
    build_bm25,
    load_embedding_model,
    search,
)
from rag_assistant import (
    load_generation_model,
    answer_query,
)


# --------------------------------------------------
# MODULE-LEVEL LAZY CACHE
# --------------------------------------------------

_retrieval_ready = False
_chunks_df = None
_faiss_index = None
_bm25 = None
_embedding_model = None

_generation_ready = False
_tokenizer = None
_gen_model = None


def _ensure_retrieval():
    """Load retrieval components once per process."""
    global _retrieval_ready, _chunks_df, _faiss_index, _bm25, _embedding_model

    if not _retrieval_ready:
        print("Initialising hybrid retrieval components …")
        _chunks_df = load_chunks()
        _faiss_index = load_faiss()
        _bm25 = build_bm25(_chunks_df)
        _embedding_model = load_embedding_model()
        _retrieval_ready = True
        print("Retrieval ready.")


def _ensure_generation():
    """Load the LLM generation model once per process."""
    global _generation_ready, _tokenizer, _gen_model

    if not _generation_ready:
        print("Initialising generation model …")
        _tokenizer, _gen_model = load_generation_model()
        _generation_ready = True
        print("Generation model ready.")


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def analyze_and_answer(
    sensor_data: dict,
    user_query: str,
    top_shap: int = 5,
) -> dict:
    """
    Full pipeline: predict → explain → retrieve → generate.

    Parameters
    ----------
    sensor_data : dict
        Mapping of all 120 sensor feature names to float values.
    user_query  : str
        Natural-language maintenance question from the operator.
    top_shap    : int
        Number of SHAP features to include in the response (default 5).

    Returns
    -------
    dict – see module docstring for the full schema.
    """

    # -------------------------------------------------
    # 1. RUL prediction
    # -------------------------------------------------
    prediction = predict(sensor_data)
    predicted_rul = prediction["predicted_rul"]
    tool_condition = prediction["tool_condition"]

    # -------------------------------------------------
    # 2. SHAP explanation
    # -------------------------------------------------
    shap_features = explain_sample(sensor_data, top_n=top_shap)

    # -------------------------------------------------
    # 3. Hybrid retrieval
    # -------------------------------------------------
    _ensure_retrieval()

    retrieved = search(
        user_query,
        _chunks_df,
        _bm25,
        _embedding_model,
        _faiss_index,
    )

    # Build a clean source list for the response
    retrieved_sources = []
    for r in retrieved[:3]:
        retrieved_sources.append({
            "source":       r["source"],
            "page":         r["page"],
            "hybrid_score": round(r["hybrid_score"], 4),
            "preview":      str(r["text"])[:300],
        })

    # -------------------------------------------------
    # 4. RAG generation — inject machine context into query
    # -------------------------------------------------
    _ensure_generation()

    # Build a context-enriched query for the RAG layer
    shap_summary = ", ".join(
        f"{s['feature']} ({s['direction']})"
        for s in shap_features[:3]
    )
    enriched_query = (
        f"{user_query}  "
        f"[Machine context: Predicted RUL={predicted_rul:.4f}, "
        f"Condition={tool_condition}, "
        f"Top SHAP drivers: {shap_summary}]"
    )

    maintenance_answer, _ = answer_query(
        query=enriched_query,
        predicted_rul=predicted_rul,
        chunks_df=_chunks_df,
        bm25=_bm25,
        embedding_model=_embedding_model,
        faiss_index=_faiss_index,
        tokenizer=_tokenizer,
        generation_model=_gen_model,
    )

    # -------------------------------------------------
    # 5. Assemble response
    # -------------------------------------------------
    return {
        "predicted_rul":     predicted_rul,
        "tool_condition":    tool_condition,
        "top_shap_features": shap_features,
        "retrieved_sources": retrieved_sources,
        "maintenance_answer": maintenance_answer,
    }


# --------------------------------------------------
# MAIN — quick smoke test
# --------------------------------------------------

def main():
    import pandas as pd

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "FeatureAndMetadata_Milling.csv"

    df = pd.read_csv(DATA_PATH, sep=";", header=1)
    df["CycleToFailureNormalized"] = (
        df["CycleToFailureNormalized"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    from predictive_maintenance import load_final_model
    _, features = load_final_model()

    sample = df[features].iloc[0].to_dict()
    query  = "What causes excessive spindle vibration?"

    print("\n" + "=" * 65)
    print("MAINTENANCE ASSISTANT — SMOKE TEST")
    print("=" * 65)

    result = analyze_and_answer(sample, query)

    print(f"\nPredicted RUL  : {result['predicted_rul']:.4f}")
    print(f"Tool Condition : {result['tool_condition']}")

    print("\nTop SHAP features:")
    for item in result["top_shap_features"]:
        print(f"  {item['feature']:<55} {item['shap_value']:+.4f}  {item['direction']}")

    print("\nRetrieved sources:")
    for s in result["retrieved_sources"]:
        print(f"  [{s['source']}] page {s['page']}  score={s['hybrid_score']}")

    print("\nMaintenance answer:")
    print(result["maintenance_answer"])


if __name__ == "__main__":
    main()
