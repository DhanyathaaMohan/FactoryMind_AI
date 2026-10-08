"""
schemas.py
----------
Pydantic request and response models for all FastAPI endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --------------------------------------------------
# SHARED SUB-MODELS
# --------------------------------------------------

class ShapFeature(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: str


class RetrievedSource(BaseModel):
    source: str
    page: int
    hybrid_score: float
    preview: str


# --------------------------------------------------
# /predict
# --------------------------------------------------

class PredictRequest(BaseModel):
    sensor_data: Dict[str, float] = Field(
        ...,
        description=(
            "Mapping of all 120 sensor feature names to their float values. "
            "Keys must match the feature names in rul_feature_names.json."
        ),
        json_schema_extra={
            "example": {
                "Accelerometer - Spindle -X - std": 0.012,
                "Accelerometer - Spindle -Z - std": 0.008,
            }
        },
    )


class PredictResponse(BaseModel):
    predicted_rul: float = Field(..., description="Normalised RUL clamped to [0, 1]")
    tool_condition: str  = Field(..., description="Healthy | Degrading | Worn | Critical")


# --------------------------------------------------
# /search
# --------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Maintenance search query")
    top_k: int = Field(3, ge=1, le=10, description="Number of results to return")


class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedSource]


# --------------------------------------------------
# /ask
# --------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Maintenance question")
    predicted_rul: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional predicted normalised RUL for context injection",
    )


class AskResponse(BaseModel):
    query: str
    tool_condition: Optional[str]
    maintenance_answer: str
    retrieved_sources: List[RetrievedSource]


# --------------------------------------------------
# /analyze-and-ask
# --------------------------------------------------

class AnalyzeRequest(BaseModel):
    sensor_data: Dict[str, float] = Field(
        ...,
        description="All 120 sensor feature values",
    )
    query: str = Field(..., min_length=1, description="Maintenance question")
    top_shap: int = Field(5, ge=1, le=10, description="Number of SHAP features to return")


class AnalyzeResponse(BaseModel):
    predicted_rul: float
    tool_condition: str
    top_shap_features: List[ShapFeature]
    retrieved_sources: List[RetrievedSource]
    maintenance_answer: str


# --------------------------------------------------
# /transcribe
# --------------------------------------------------

class TranscribeResponse(BaseModel):
    filename: str
    transcript: str


# --------------------------------------------------
# Health check
# --------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    retrieval_loaded: bool
    generation_loaded: bool
