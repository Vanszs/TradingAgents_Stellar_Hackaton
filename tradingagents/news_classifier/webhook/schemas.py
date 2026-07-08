"""Pydantic schemas for the news classifier webhook."""

from pydantic import BaseModel, Field
from typing import Optional


class ClassifyRequest(BaseModel):
    title: str = Field(..., description="News article title")
    content: str = Field("", description="News article content")
    source: str = Field("", description="News source URL or name")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Classification: CRITICAL, MODERATE, or NORMAL")
    confidence: float = Field(..., description="Confidence score 0-1")
    probabilities: dict[str, float] = Field(..., description="Probabilities per class")
    title: str
    source: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str = "0.2.0"
