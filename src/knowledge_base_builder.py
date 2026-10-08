import json
import re

from pathlib import Path

import numpy as np
import pandas as pd
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_DIR = (
    BASE_DIR
    / "knowledge_base"
    / "documents"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "knowledge_base"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CHUNK_SIZE = 700

CHUNK_OVERLAP = 120

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    return text


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf(pdf_path):

    print(
        "\nReading PDF:",
        pdf_path.name
    )

    reader = PdfReader(
        pdf_path
    )

    extracted_pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text()

        except Exception as error:

            print(
                f"Could not read page "
                f"{page_number}: {error}"
            )

            continue

        text = clean_text(
            text
        )

        if text:

            extracted_pages.append({

                "source":
                    pdf_path.name,

                "page":
                    page_number,

                "text":
                    text
            })

    print(
        "Pages extracted:",
        len(extracted_pages)
    )

    return extracted_pages


# --------------------------------------------------
# TXT EXTRACTION
# --------------------------------------------------

def extract_txt(txt_path):

    print(
        "\nReading TXT:",
        txt_path.name
    )

    try:

        text = txt_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        text = txt_path.read_text(
            encoding="latin-1"
        )

    text = clean_text(
        text
    )

    if not text:

        return []

    return [{

        "source":
            txt_path.name,

        "page":
            1,

        "text":
            text
    }]


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

def load_documents():

    print("\n")
    print("=" * 60)
    print("LOADING MAINTENANCE DOCUMENTS")
    print("=" * 60)

    if not DOCUMENT_DIR.exists():

        DOCUMENT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "\nDocument directory created:"
        )

        print(
            DOCUMENT_DIR
        )

        print(
            "\nAdd maintenance PDFs or TXT "
            "files and run again."
        )

        return []

    files = list(
        DOCUMENT_DIR.iterdir()
    )

    supported_files = [

        file
        for file in files

        if file.suffix.lower()
        in [".pdf", ".txt"]
    ]

    print(
        "\nSupported documents found:",
        len(supported_files)
    )

    documents = []

    for file_path in supported_files:

        extension = (
            file_path
            .suffix
            .lower()
        )

        if extension == ".pdf":

            documents.extend(
                extract_pdf(
                    file_path
                )
            )

        elif extension == ".txt":

            documents.extend(
                extract_txt(
                    file_path
                )
            )

    print(
        "\nTotal extracted document sections:",
        len(documents)
    )

    return documents


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_words = (
            words[start:end]
        )

        chunk = " ".join(
            chunk_words
        )

        if chunk.strip():

            chunks.append(
                chunk
            )

        if end >= len(words):
            break

        start = (
            end - overlap
        )

    return chunks


# --------------------------------------------------
# CREATE KNOWLEDGE CHUNKS
# --------------------------------------------------

def create_chunks(
    documents
):

    print("\n")
    print("=" * 60)
    print("CREATING TEXT CHUNKS")
    print("=" * 60)

    chunk_records = []

    chunk_id = 0

    for document in documents:

        text_chunks = split_text(
            document["text"]
        )

        for chunk_index, chunk in enumerate(
            text_chunks,
            start=1
        ):

            chunk_records.append({

                "chunk_id":
                    chunk_id,

                "source":
                    document["source"],

                "page":
                    document["page"],

                "chunk_number":
                    chunk_index,

                "text":
                    chunk
            })

            chunk_id += 1

    chunks_df = pd.DataFrame(
        chunk_records
    )

    print(
        "\nTotal chunks created:",
        len(chunks_df)
    )

    if not chunks_df.empty:

        print(
            "\nDocuments represented:"
        )

        print(
            chunks_df[
                "source"
            ]
            .value_counts()
        )

    return chunks_df


# --------------------------------------------------
# SAVE CHUNKS
# --------------------------------------------------

def save_chunks(
    chunks_df
):

    csv_path = (
        OUTPUT_DIR
        / "maintenance_chunks.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "maintenance_chunks.json"
    )

    chunks_df.to_csv(
        csv_path,
        index=False
    )

    records = (
        chunks_df
        .to_dict(
            orient="records"
        )
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nChunks CSV saved:",
        csv_path
    )

    print(
        "Chunks JSON saved:",
        json_path
    )


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

def load_embedding_model():

    print("\n")
    print("=" * 60)
    print("LOADING SENTENCE TRANSFORMER")
    print("=" * 60)

    print(
        "\nModel:",
        EMBEDDING_MODEL_NAME
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    return model


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

def create_embeddings(
    chunks_df,
    model
):

    print("\n")
    print("=" * 60)
    print("CREATING EMBEDDINGS")
    print("=" * 60)

    texts = (
        chunks_df[
            "text"
        ]
        .tolist()
    )

    embeddings = model.encode(

        texts,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    print(
        "\nEmbedding matrix shape:",
        embeddings.shape
    )

    return embeddings


# --------------------------------------------------
# SAVE EMBEDDINGS
# --------------------------------------------------

def save_embeddings(
    embeddings
):

    path = (
        OUTPUT_DIR
        / "maintenance_embeddings.npy"
    )

    np.save(
        path,
        embeddings
    )

    print(
        "Embeddings saved:",
        path
    )


# --------------------------------------------------
# BUILD FAISS INDEX
# --------------------------------------------------

def build_faiss_index(
    embeddings
):

    print("\n")
    print("=" * 60)
    print("BUILDING FAISS INDEX")
    print("=" * 60)

    dimension = (
        embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        "\nFAISS vectors stored:",
        index.ntotal
    )

    index_path = (
        OUTPUT_DIR
        / "maintenance_faiss.index"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    print(
        "FAISS index saved:",
        index_path
    )

    return index


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    documents = (
        load_documents()
    )

    if len(documents) == 0:

        print(
            "\nNo usable maintenance "
            "documents were found."
        )

        print(
            "\nAdd PDF/TXT files inside:"
        )

        print(
            DOCUMENT_DIR
        )

        return

    chunks_df = (
        create_chunks(
            documents
        )
    )

    if chunks_df.empty:

        print(
            "\nNo text chunks could "
            "be created."
        )

        return

    save_chunks(
        chunks_df
    )

    embedding_model = (
        load_embedding_model()
    )

    embeddings = (
        create_embeddings(
            chunks_df,
            embedding_model
        )
    )

    save_embeddings(
        embeddings
    )

    build_faiss_index(
        embeddings
    )

    print("\n")
    print("=" * 60)
    print("KNOWLEDGE BASE BUILD COMPLETED")
    print("=" * 60)

    print(
        "\nGenerated files:"
    )

    print(
        "1. maintenance_chunks.csv"
    )

    print(
        "2. maintenance_chunks.json"
    )

    print(
        "3. maintenance_embeddings.npy"
    )

    print(
        "4. maintenance_faiss.index"
    )


if __name__ == "__main__":
    main()