import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from hybrid_retrieval import (
    load_chunks,
    load_faiss,
    build_bm25,
    load_embedding_model,
    search
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

GENERATION_MODEL_NAME = "google/flan-t5-small"

RETRIEVAL_TOP_K = 3


# --------------------------------------------------
# TOOL CONDITION FROM RUL
# --------------------------------------------------

def get_tool_condition(rul):

    if rul is None:
        return "Unknown"

    if rul >= 0.75:
        return "Healthy"

    elif rul >= 0.50:
        return "Degrading"

    elif rul >= 0.25:
        return "Worn"

    else:
        return "Critical"


# --------------------------------------------------
# LOAD GENERATION MODEL
# --------------------------------------------------

def load_generation_model():

    print("\n")
    print("=" * 65)
    print("LOADING LOCAL GENERATION MODEL")
    print("=" * 65)

    print(
        "\nModel:",
        GENERATION_MODEL_NAME
    )

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATION_MODEL_NAME
    )

    model.eval()

    print(
        "Generation model loaded."
    )

    return tokenizer, model


# --------------------------------------------------
# PREPARE RETRIEVED CONTEXT
# --------------------------------------------------

def prepare_context(
    retrieved_results
):

    context_parts = []

    for rank, result in enumerate(
        retrieved_results[:RETRIEVAL_TOP_K],
        start=1
    ):

        text = str(
            result["text"]
        )

        # Keep context compact for local model
        text = text[:1600]

        context = (
            f"Document {rank}\n"
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Content: {text}"
        )

        context_parts.append(
            context
        )

    return "\n\n".join(
        context_parts
    )


# --------------------------------------------------
# BUILD RAG PROMPT
# --------------------------------------------------

def build_prompt(
    query,
    context,
    predicted_rul=None
):

    tool_condition = (
        get_tool_condition(
            predicted_rul
        )
    )

    if predicted_rul is None:

        machine_context = (
            "Predicted RUL: Not available\n"
            "Tool Condition: Not available"
        )

    else:

        machine_context = (
            f"Predicted Normalized RUL: "
            f"{predicted_rul:.4f}\n"
            f"Tool Condition: "
            f"{tool_condition}"
        )

    prompt = f"""
You are a CNC milling maintenance assistant.

Answer the maintenance question using only the
retrieved technical information below.

Do not invent technical facts.

If the retrieved information is insufficient,
say that the available maintenance documents
do not provide enough information.

Machine Condition:
{machine_context}

User Question:
{query}

Retrieved Maintenance Information:
{context}

Provide a concise maintenance response containing:

1. Likely causes
2. Recommended checks or corrective actions
3. How the current machine condition should
   influence maintenance urgency

Answer:
"""

    return prompt


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

def generate_answer(
    prompt,
    tokenizer,
    model
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.2
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()


# --------------------------------------------------
# DISPLAY SOURCES
# --------------------------------------------------

def display_sources(
    retrieved_results
):

    print("\n")
    print("=" * 65)
    print("RETRIEVED SOURCES")
    print("=" * 65)

    for rank, result in enumerate(
        retrieved_results[:RETRIEVAL_TOP_K],
        start=1
    ):

        print(
            f"\n[{rank}] "
            f"{result['source']}"
        )

        print(
            "Page:",
            result["page"]
        )

        print(
            "Hybrid score:",
            round(
                result["hybrid_score"],
                4
            )
        )


# --------------------------------------------------
# RAG QUERY
# --------------------------------------------------

def answer_query(
    query,
    predicted_rul,
    chunks_df,
    bm25,
    embedding_model,
    faiss_index,
    tokenizer,
    generation_model
):

    retrieved_results = search(
        query,
        chunks_df,
        bm25,
        embedding_model,
        faiss_index
    )

    context = prepare_context(
        retrieved_results
    )

    prompt = build_prompt(
        query,
        context,
        predicted_rul
    )

    answer = generate_answer(
        prompt,
        tokenizer,
        generation_model
    )

    return (
        answer,
        retrieved_results
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("CNC MAINTENANCE RAG ASSISTANT")
    print("=" * 70)

    # -----------------------------
    # Load retrieval components
    # -----------------------------

    chunks_df = load_chunks()

    faiss_index = load_faiss()

    bm25 = build_bm25(
        chunks_df
    )

    embedding_model = (
        load_embedding_model()
    )

    # -----------------------------
    # Load generation model
    # -----------------------------

    tokenizer, generation_model = (
        load_generation_model()
    )

    print("\n")
    print("=" * 70)
    print("SYSTEM READY")
    print("=" * 70)

    print(
        "\nEnter 'exit' to stop."
    )

    while True:

        query = input(
            "\nEnter maintenance question: "
        ).strip()

        if query.lower() == "exit":

            print(
                "\nRAG assistant stopped."
            )

            break

        if not query:

            print(
                "Please enter a question."
            )

            continue

        # ---------------------------------
        # Manual RUL input for current test
        # ---------------------------------

        rul_input = input(
            "Enter predicted normalized RUL "
            "(0 to 1, or press Enter if unavailable): "
        ).strip()

        if rul_input == "":

            predicted_rul = None

        else:

            try:

                predicted_rul = float(
                    rul_input
                )

                if (
                    predicted_rul < 0
                    or predicted_rul > 1
                ):

                    print(
                        "RUL should be between 0 and 1."
                    )

                    continue

            except ValueError:

                print(
                    "Invalid RUL value."
                )

                continue

        tool_condition = (
            get_tool_condition(
                predicted_rul
            )
        )

        print("\n")
        print("-" * 70)

        if predicted_rul is not None:

            print(
                "Machine Context"
            )

            print(
                "Predicted RUL:",
                round(
                    predicted_rul,
                    4
                )
            )

            print(
                "Tool Condition:",
                tool_condition
            )

        else:

            print(
                "Machine context not provided."
            )

        print("-" * 70)

        # ---------------------------------
        # RAG generation
        # ---------------------------------

        answer, retrieved_results = (
            answer_query(
                query,
                predicted_rul,
                chunks_df,
                bm25,
                embedding_model,
                faiss_index,
                tokenizer,
                generation_model
            )
        )

        print("\n")
        print("=" * 70)
        print("RAG MAINTENANCE RESPONSE")
        print("=" * 70)

        print(
            "\n",
            answer
        )

        display_sources(
            retrieved_results
        )


if __name__ == "__main__":
    main()