# FactoryMind AI

**An Intelligent Multimodal Predictive Maintenance and Retrieval-Augmented Knowledge Assistance System for CNC Milling Machines**

Final-year project — Machine Learning &amp; AI Engineering

---

## Project Overview

FactoryMind AI combines machine-learning-based predictive maintenance with a Retrieval-Augmented Generation (RAG) knowledge assistant to help operators monitor CNC milling machine health, predict tool remaining useful life (RUL), and retrieve grounded maintenance guidance from Haas technical documentation.

The system integrates:

- **Predictive RUL model** — Random Forest regressor trained on 120 CNC sensor features
- **SHAP explainability** — per-prediction feature-influence explanations
- **Hybrid retrieval** — BM25 + FAISS semantic search over Haas manuals
- **RAG assistant** — grounded maintenance answers generated from retrieved document chunks
- **Voice input** — audio transcription via Faster-Whisper (CPU)
- **FastAPI backend** — REST API serving all pipeline components
- **React dashboard** — industrial-style web UI

---

## Dataset

| Property | Value |
|---|---|
| File | `data/FeatureAndMetadata_Milling.csv` |
| Rows | 968 |
| Columns | 131 |
| Tools | 14 (Tool 11 excluded from cross-validation — only 1 sample) |
| Accelerometer features | 48 |
| Current features | 72 |
| Total sensor features | 120 |
| Regression target | `CycleToFailureNormalized` (normalised to [0, 1]) |
| Group column | `TollIndex` (used only for grouping, not as a model input) |

**Dataset source:** UC Berkeley milling dataset (public domain), pre-processed with engineered statistical features (mean, std, kurtosis, skew, max) computed per machining cycle.

Columns excluded from model input: `NumberOfCycle`, `CycleToFailure`, `CycleToFailureNormalized`, `TollIndex`.

---

## System Architecture

```
CNC Sensor Features (120)
         │
         ▼
 Random Forest Regressor
         │
         ▼
  Normalised RUL ──────────────┐
         │                     │
         ▼                     ▼
 Tool Condition Map      SHAP Explainer
 (threshold-based)      (top-N features)
         │
         └─────────────────────┐
                               │
User Text / Voice Query        │
         │                     │
         ▼                     │
  Faster-Whisper (CPU)         │
         │                     │
         ▼                     │
 BM25 + FAISS Hybrid           │
      Retrieval                │
         │                     │
         ▼                     │
 Relevant Haas Chunks  ◄───────┘
         │
         ▼
  Flan-T5 RAG Assistant
  (context-grounded)
         │
         ▼
    FastAPI Backend
         │
         ▼
  React Dashboard
```

---

## ML Results

### Model Selection — 5-fold GroupKFold Cross-Validation

Groups are defined by `TollIndex` so no tool appears in both training and test folds.

| Model | MAE | RMSE | R² (avg) | R² Std |
|---|---|---|---|---|
| **Random Forest** ✓ | **0.1740** | **0.2166** | **0.4130** | 0.3128 |
| Extra Trees | 0.1782 | 0.2181 | 0.3924 | 0.3653 |
| XGBoost | 0.1814 | 0.2288 | 0.3380 | 0.3908 |

Random Forest was selected as the final model. The relatively high R² standard deviation (0.31) reflects genuine tool-to-tool variability in the small dataset — some folds contain tools with fewer training examples.

**Important:** R² is reported as a regression quality measure. It is not a classification accuracy and should not be interpreted as one.

### Per-Fold Cross-Validation Results

| Fold | Test Tools | MAE | RMSE | R² |
|---|---|---|---|---|
| 1 | 3, 9 | 0.2442 | 0.2964 | −0.065 |
| 2 | 2, 6, 105 | 0.1171 | 0.1453 | 0.753 |
| 3 | 5, 7, 10 | 0.1393 | 0.1799 | 0.621 |
| 4 | 8, 103 | 0.1950 | 0.2365 | 0.344 |
| 5 | 4, 101, 102 | 0.1743 | 0.2247 | 0.413 |

---

## Feature Selection

Five feature-set sizes were evaluated (120 / 60 / 30 / 20 / 10). The full set of **120 sensor features** produced the best cross-validated performance and is used in the final model.

---

## SHAP Explainability

SHAP TreeExplainer was applied to the final Random Forest using 300 randomly sampled rows. SHAP values quantify each feature's **contribution to a specific model prediction**. They reflect model influence, not physical causality.

**Top 10 global features (mean |SHAP|):**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Accelerometer - Spindle -X - std | 0.1711 |
| 2 | Accelerometer - Spindle -Z - std | 0.0171 |
| 3 | Current - Driving axle X L2 - mean | 0.0165 |
| 4 | Current - Driving axle X L1 - mean | 0.0162 |
| 5 | Accelerometer - Spindle +Y - kurtosis | 0.0158 |
| 6 | Accelerometer - X Driving axle -X - std | 0.0152 |
| 7 | Accelerometer - Y Driving axle -X - std | 0.0104 |
| 8 | Accelerometer - Y Driving axle +Y - std | 0.0104 |
| 9 | Accelerometer - Spindle -Z - max | 0.0099 |
| 10 | Accelerometer - X Driving axle +Z - skew | 0.0086 |

---

## Tool Condition Strategy

A separate Random Forest classifier for tool condition was tested but achieved only **48.9% accuracy** and **0.457 macro-F1** — insufficient for reliable deployment.

Tool condition is therefore **derived directly from the predicted normalised RUL** using fixed thresholds:

| Predicted RUL | Condition |
|---|---|
| ≥ 0.75 | Healthy |
| 0.50 – 0.74 | Degrading |
| 0.25 – 0.49 | Worn |
| < 0.25 | Critical |

---

## Knowledge Base

| Property | Value |
|---|---|
| Documents | `haas_mill_operator_manual.pdf`, `VF_VM - 40T - Spindle - Haas Service Manual.pdf` |
| Extracted sections | 612 |
| Chunks | 629 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| FAISS vectors | 629 |

Chunks are stored in `outputs/knowledge_base/`.

---

## BM25 + FAISS Hybrid Retrieval

Queries are processed by two complementary retrievers whose normalised scores are combined:

```
hybrid_score = 0.35 × BM25_score + 0.65 × FAISS_score
```

- **BM25** (rank-bm25) — keyword matching, effective for exact maintenance terms
- **FAISS** (faiss-cpu, IndexFlatIP) — semantic similarity via MiniLM-L6-v2 embeddings

The top-3 highest-scoring chunks are injected into the RAG prompt.

---

## RAG Workflow

1. User submits a maintenance query (text or voice)
2. Hybrid retriever fetches the top-3 most relevant Haas document chunks
3. Machine context is assembled: predicted RUL, tool condition, top SHAP drivers
4. A structured prompt is constructed containing the retrieved chunks and machine context
5. `google/flan-t5-small` generates a grounded maintenance recommendation
6. The response includes: **Likely Causes**, **Recommended Checks**, **Maintenance Urgency**, and **Sources** (document name + page)

The generation model answers **only from retrieved content**. If the retrieved chunks do not contain relevant information, the model explicitly says so rather than inventing facts.

**Note:** RAG response quality has not been formally evaluated with an automated metric (e.g. RAGAS). Qualitative testing confirmed that retrieved chunks are topically relevant for spindle-vibration and lubrication queries.

---

## Installation

### Prerequisites

- Python 3.11 or 3.13
- Node.js 18+
- Git

### Python environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Train the final model

Run once to train the Random Forest on all 968 samples and save artefacts to `models/`:

```bash
python src/predictive_maintenance.py
```

This produces:
- `models/rul_random_forest_final.pkl`
- `models/rul_feature_names.json`

### Build the knowledge base (already done)

If the `outputs/knowledge_base/` files are missing, rebuild with:

```bash
python src/knowledge_base_builder.py
```

---

## Backend

Start the FastAPI server:

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive API docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | System component status |
| POST | `/predict` | RUL prediction + tool condition |
| POST | `/search` | Hybrid knowledge-base search |
| POST | `/ask` | RAG maintenance answer |
| POST | `/analyze-and-ask` | Full pipeline (predict + explain + retrieve + generate) |
| POST | `/transcribe` | Audio → transcript (Faster-Whisper) |

---

## Frontend

```bash
cd frontend
npm install
npm run dev        # development server on http://localhost:5173
npm run build      # production build → frontend/dist/
```

The development server proxies `/api/*` requests to the FastAPI backend on port 8000.

---

## Tests

```bash
# Activate the venv first
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

pytest
```

Tests require the final model artefacts (`models/rul_random_forest_final.pkl`) and the knowledge-base files. Modules that depend on missing artefacts are skipped automatically.

Test coverage:
- `test_condition_mapping.py` — 15 threshold boundary tests
- `test_retrieval.py` — 7 test types × 5 maintenance queries
- `test_backend.py` — GET /health, POST /predict, POST /search (5 queries each), POST /ask

---

## Limitations

- The dataset is small (968 rows, 14 tools). Cross-validated R² ranges from −0.065 to 0.753 depending on which tools appear in the test fold — generalisation to unseen machine configurations is limited.
- The RAG generation model (`flan-t5-small`) is a small encoder-decoder. Answers can be incomplete or structurally inconsistent for complex multi-step queries.
- RAG retrieval quality has not been evaluated with a formal metric (e.g. RAGAS, BERTScore).
- Voice transcription requires an audio file upload — real-time microphone streaming is not implemented.
- The frontend `/analyze-and-ask` panel sends placeholder sensor values (zeros). A production deployment would need live sensor integration.
- All inference runs on CPU; latency for model loading and SHAP computation is noticeable on first request.

---

## Future Work

- Integrate live OPC-UA or MTConnect sensor streaming into the dashboard
- Replace `flan-t5-small` with a larger instruction-tuned LLM (e.g. Mistral-7B via llama.cpp) for higher-quality maintenance answers
- Add RAGAS evaluation to benchmark RAG retrieval and answer faithfulness
- Extend the dataset with more tools and machining conditions to improve cross-tool generalisation
- Add sensor data entry form to the frontend so operators can submit real readings
- Implement real-time microphone streaming for voice queries
- Add authentication and operator role management
- Containerise with Docker for easier deployment
- Expand the knowledge base with additional Haas service bulletins and maintenance schedules

---

## Project Structure

```
FactoryMind AI/
├── data/
│   └── FeatureAndMetadata_Milling.csv
├── knowledge_base/
│   └── documents/
│       ├── haas_mill_operator_manual.pdf
│       └── VF_VM - 40T - Spindle - Haas Service Manual.pdf
├── models/
│   ├── rul_random_forest_final.pkl
│   └── rul_feature_names.json
├── outputs/
│   ├── knowledge_base/       (chunks, embeddings, FAISS index)
│   ├── plots/                (SHAP plots, RUL plots)
│   └── results/              (CSV evaluation results)
├── src/
│   ├── dataset_exploration.py
│   ├── feature_analysis.py
│   ├── feature_selection_rul.py
│   ├── rul_model.py
│   ├── rul_validation.py
│   ├── model_comparison.py
│   ├── shap_rul.py
│   ├── tool_condition_classification.py
│   ├── knowledge_base_builder.py
│   ├── hybrid_retrieval.py
│   ├── rag_assistant.py
│   ├── predictive_maintenance.py
│   ├── explainability.py
│   ├── maintenance_assistant.py
│   └── voice_search.py
├── backend/
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── prediction_service.py
│       ├── retrieval_service.py
│       └── rag_service.py
├── frontend/
│   ├── src/
│   │   ├── components/       (Header, MachineHealthCard, RULGauge, ...)
│   │   ├── pages/            (Dashboard)
│   │   └── services/         (api.js)
│   └── package.json
├── tests/
│   ├── test_condition_mapping.py
│   ├── test_retrieval.py
│   └── test_backend.py
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## License

This project is submitted as academic work. The Haas operator and service manuals included in the knowledge base are the property of Haas Automation, Inc. and are used here for academic research purposes only.
