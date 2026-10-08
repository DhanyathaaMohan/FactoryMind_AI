import re
import numpy as np
import pandas as pd
import faiss

from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

KB_DIR = (
    BASE_DIR
    / "outputs"
    / "knowledge_base"
)

CHUNKS_PATH = (
    KB_DIR
    / "maintenance_chunks.csv"
)

FAISS_PATH = (
    KB_DIR
    / "maintenance_faiss.index"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K_BM25 = 10

TOP_K_FAISS = 10

FINAL_TOP_K = 5

BM25_WEIGHT = 0.35

FAISS_WEIGHT = 0.65


# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

def tokenize(text):

    text = text.lower()

    tokens = re.findall(
        r"\b\w+\b",
        text
    )

    return tokens


# --------------------------------------------------
# LOAD KNOWLEDGE BASE
# --------------------------------------------------

def load_chunks():

    chunks_df = pd.read_csv(
        CHUNKS_PATH
    )

    print(
        "\nChunks loaded:",
        len(chunks_df)
    )

    return chunks_df


def load_faiss():

    index = faiss.read_index(
        str(FAISS_PATH)
    )

    print(
        "FAISS vectors loaded:",
        index.ntotal
    )

    return index


# --------------------------------------------------
# BUILD BM25
# --------------------------------------------------

def build_bm25(chunks_df):

    print(
        "\nBuilding BM25 index..."
    )

    tokenized_corpus = [

        tokenize(text)

        for text in chunks_df[
            "text"
        ].astype(str)

    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    print(
        "BM25 index ready."
    )

    return bm25


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

def load_embedding_model():

    print(
        "\nLoading Sentence Transformer..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        "Embedding model ready."
    )

    return model


# --------------------------------------------------
# NORMALIZE SCORES
# --------------------------------------------------

def normalize_scores(scores):

    scores = np.array(
        scores,
        dtype=float
    )

    minimum = scores.min()

    maximum = scores.max()

    if maximum == minimum:

        return np.zeros_like(
            scores
        )

    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def bm25_search(
    query,
    bm25,
    chunks_df,
    top_k=TOP_K_BM25
):

    query_tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    selected_scores = scores[
        top_indices
    ]

    normalized = normalize_scores(
        selected_scores
    )

    for index, score, norm_score in zip(
        top_indices,
        selected_scores,
        normalized
    ):

        row = chunks_df.iloc[
            index
        ]

        results.append({

            "chunk_id":
                int(row["chunk_id"]),

            "source":
                row["source"],

            "page":
                int(row["page"]),

            "text":
                row["text"],

            "bm25_score":
                float(score),

            "bm25_normalized":
                float(norm_score)
        })

    return results


# --------------------------------------------------
# FAISS SEARCH
# --------------------------------------------------

def faiss_search(
    query,
    model,
    index,
    chunks_df,
    top_k=TOP_K_FAISS
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = (
        query_embedding
        .astype(
            "float32"
        )
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    raw_scores = scores[0]

    normalized = normalize_scores(
        raw_scores
    )

    for idx, score, norm_score in zip(
        indices[0],
        raw_scores,
        normalized
    ):

        if idx == -1:
            continue

        row = chunks_df.iloc[
            idx
        ]

        results.append({

            "chunk_id":
                int(row["chunk_id"]),

            "source":
                row["source"],

            "page":
                int(row["page"]),

            "text":
                row["text"],

            "faiss_score":
                float(score),

            "faiss_normalized":
                float(norm_score)
        })

    return results


# --------------------------------------------------
# HYBRID FUSION
# --------------------------------------------------

def combine_results(
    bm25_results,
    faiss_results
):

    combined = {}

    for result in bm25_results:

        chunk_id = result[
            "chunk_id"
        ]

        combined[
            chunk_id
        ] = {

            "chunk_id":
                chunk_id,

            "source":
                result["source"],

            "page":
                result["page"],

            "text":
                result["text"],

            "bm25_score":
                result[
                    "bm25_normalized"
                ],

            "faiss_score":
                0.0
        }

    for result in faiss_results:

        chunk_id = result[
            "chunk_id"
        ]

        if chunk_id not in combined:

            combined[
                chunk_id
            ] = {

                "chunk_id":
                    chunk_id,

                "source":
                    result["source"],

                "page":
                    result["page"],

                "text":
                    result["text"],

                "bm25_score":
                    0.0,

                "faiss_score":
                    result[
                        "faiss_normalized"
                    ]
            }

        else:

            combined[
                chunk_id
            ][
                "faiss_score"
            ] = result[
                "faiss_normalized"
            ]

    final_results = []

    for result in combined.values():

        hybrid_score = (

            BM25_WEIGHT
            * result[
                "bm25_score"
            ]

            +

            FAISS_WEIGHT
            * result[
                "faiss_score"
            ]
        )

        result[
            "hybrid_score"
        ] = hybrid_score

        final_results.append(
            result
        )

    final_results.sort(
        key=lambda x:
            x["hybrid_score"],
        reverse=True
    )

    return final_results[
        :FINAL_TOP_K
    ]


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def display_results(
    query,
    results
):

    print("\n")
    print("=" * 75)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 75)

    print(
        "\nQuery:",
        query
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print("\n")
        print("-" * 75)

        print(
            f"Rank {rank}"
        )

        print(
            "Source:",
            result["source"]
        )

        print(
            "Page:",
            result["page"]
        )

        print(
            "BM25 score:",
            round(
                result[
                    "bm25_score"
                ],
                4
            )
        )

        print(
            "FAISS score:",
            round(
                result[
                    "faiss_score"
                ],
                4
            )
        )

        print(
            "Hybrid score:",
            round(
                result[
                    "hybrid_score"
                ],
                4
            )
        )

        print(
            "\nText:"
        )

        print(
            result["text"]
        )


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def search(
    query,
    chunks_df,
    bm25,
    model,
    faiss_index
):

    bm25_results = bm25_search(
        query,
        bm25,
        chunks_df
    )

    faiss_results = faiss_search(
        query,
        model,
        faiss_index,
        chunks_df
    )

    hybrid_results = combine_results(
        bm25_results,
        faiss_results
    )

    return hybrid_results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    chunks_df = load_chunks()

    faiss_index = load_faiss()

    bm25 = build_bm25(
        chunks_df
    )

    embedding_model = (
        load_embedding_model()
    )

    print("\n")
    print("=" * 75)
    print("HYBRID MAINTENANCE SEARCH")
    print("=" * 75)

    print(
        "\nType 'exit' to stop."
    )

    while True:

        query = input(
            "\nEnter maintenance query: "
        ).strip()

        if query.lower() == "exit":

            print(
                "\nSearch terminated."
            )

            break

        if not query:

            print(
                "Please enter a query."
            )

            continue

        results = search(
            query,
            chunks_df,
            bm25,
            embedding_model,
            faiss_index
        )

        display_results(
            query,
            results
        )


if __name__ == "__main__":
    main()