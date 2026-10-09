"""
main.py
-------
FactoryMind FastAPI backend.

Start with:
    uvicorn backend.main:app --reload

Endpoints
---------
GET  /                    – welcome message
GET  /health              – system health and component status
POST /predict             – RUL prediction + tool condition
POST /search              – hybrid knowledge-base search
POST /ask                 – RAG maintenance answer (no sensor data required)
POST /analyze-and-ask     – full pipeline: predict + explain + retrieve + generate
POST /transcribe          – audio → transcript via Faster-Whisper
"""

import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Add src/ to sys.path so all project modules resolve correctly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backend.schemas import (
    PredictRequest,
    PredictResponse,
    SearchRequest,
    SearchResponse,
    AskRequest,
    AskResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    TranscribeResponse,
    HealthResponse,
    RetrievedSource,
    ShapFeature,
    SampleMetadata,
    SamplesResponse,
)
from backend.services import prediction_service, retrieval_service, rag_service


# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = FastAPI(
    title="FactoryMind AI",
    description=(
        "Intelligent Multimodal Predictive Maintenance and "
        "Retrieval-Augmented Knowledge Assistance for CNC Milling Machines"
    ),
    version="1.0.0",
)

# Allow the React dev server (port 5173) and any origin in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/", tags=["General"])
def root():
    return {
        "project": "FactoryMind AI",
        "description": (
            "Predictive maintenance and RAG knowledge assistant "
            "for CNC milling machines"
        ),
        "docs": "/docs",
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=prediction_service.check_model_available(),
        retrieval_loaded=retrieval_service.is_loaded(),
        generation_loaded=rag_service.is_generation_loaded(),
    )


# --------------------------------------------------
# SAMPLES
# --------------------------------------------------

@app.get("/samples", response_model=SamplesResponse, tags=["Dataset"])
def list_samples():
    """
    Return lightweight metadata for all dataset samples.
    Use /samples/{index} to fetch features for a specific sample.
    """
    try:
        return prediction_service.get_samples_metadata()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Dataset not available")


# --------------------------------------------------
# SAMPLES BY INDEX
# --------------------------------------------------

@app.get("/samples/{sample_index}", tags=["Dataset"])
def get_sample(sample_index: int):
    """
    Return metadata for a specific sample by its dataset index.
    """
    try:
        return prediction_service.get_sample_metadata(sample_index)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Sample index {sample_index} not found")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Dataset not available")


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict the normalised Remaining Useful Life (RUL) for a set of
    CNC sensor readings and derive the current tool condition.
    """
    try:
        result = prediction_service.run_prediction(request.sensor_data)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictResponse(
        predicted_rul=result["predicted_rul"],
        tool_condition=result["tool_condition"],
    )


# --------------------------------------------------
# SEARCH
# --------------------------------------------------

@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
def search(request: SearchRequest):
    """
    Hybrid BM25 + FAISS search over the Haas maintenance knowledge base.
    Returns ranked document chunks with source, page and preview.
    """
    try:
        results = retrieval_service.search(request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        RetrievedSource(
            source=r["source"],
            page=r["page"],
            hybrid_score=r["hybrid_score"],
            preview=r["preview"],
        )
        for r in results
    ]
    return SearchResponse(query=request.query, results=sources)


# --------------------------------------------------
# ASK
# --------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest):
    """
    Answer a maintenance question using retrieved Haas documentation.
    Optionally accepts a predicted RUL value to enrich the prompt.
    """
    try:
        answer, sources = rag_service.ask(
            query=request.query,
            predicted_rul=request.predicted_rul,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    tool_condition = None
    if request.predicted_rul is not None:
        from predictive_maintenance import map_tool_condition
        tool_condition = map_tool_condition(request.predicted_rul)

    return AskResponse(
        query=request.query,
        tool_condition=tool_condition,
        maintenance_answer=answer,
        retrieved_sources=[
            RetrievedSource(**s) for s in sources
        ],
    )


# --------------------------------------------------
# ANALYZE AND ASK
# --------------------------------------------------

@app.post("/analyze-and-ask", response_model=AnalyzeResponse, tags=["Full Pipeline"])
def analyze_and_ask(request: AnalyzeRequest):
    """
    Full pipeline:
      1. Predict RUL + tool condition from sensor data
      2. Compute SHAP feature contributions
      3. Retrieve relevant Haas maintenance chunks
      4. Generate a grounded maintenance recommendation

    Accepts either:
    - sample_index: loads sensor data from the dataset
    - sensor_data: direct feature dictionary (must contain 120 features)
    """
    try:
        # Determine which data source to use
        if request.sample_index is not None and request.sensor_data is None:
            # Load from dataset
            sensor_data = prediction_service.get_sample_sensor_data(request.sample_index)
        elif request.sample_index is None and request.sensor_data is not None:
            # Use provided sensor data
            sensor_data = request.sensor_data
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either sample_index OR sensor_data, not both"
            )

        # Validate sensor_data
        if sensor_data is None or len(sensor_data) == 0:
            raise HTTPException(
                status_code=422,
                detail="Sensor data is empty. Ensure sample_index is valid or sensor_data contains features."
            )

        # Prediction
        pred = prediction_service.run_prediction(sensor_data)

        # SHAP explanation
        shap_feats = prediction_service.run_explanation(
            sensor_data, top_n=request.top_shap
        )

        # Enrich query with machine context
        shap_summary = ", ".join(
            f"{s['feature']} ({s['direction']})" for s in shap_feats[:3]
        )
        enriched_query = (
            f"{request.query}  "
            f"[Machine context: Predicted RUL={pred['predicted_rul']:.4f}, "
            f"Condition={pred['tool_condition']}, "
            f"Top drivers: {shap_summary}]"
        )

        # RAG answer
        answer, sources = rag_service.ask(
            query=enriched_query,
            predicted_rul=pred["predicted_rul"],
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AnalyzeResponse(
        predicted_rul=pred["predicted_rul"],
        tool_condition=pred["tool_condition"],
        top_shap_features=[ShapFeature(**s) for s in shap_feats],
        retrieved_sources=[RetrievedSource(**s) for s in sources],
        maintenance_answer=answer,
    )


# --------------------------------------------------
# TRANSCRIBE
# --------------------------------------------------

@app.post("/transcribe", response_model=TranscribeResponse, tags=["Voice"])
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file (wav/mp3/m4a/ogg/flac/webm)
    using Faster-Whisper on CPU.
    """
    from voice_search import transcribe as whisper_transcribe, SUPPORTED_EXTENSIONS

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported audio format '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # Write the uploaded bytes to a temp file so Whisper can read it
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        transcript = whisper_transcribe(tmp_path)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    return TranscribeResponse(
        filename=file.filename,
        transcript=transcript,
    )
