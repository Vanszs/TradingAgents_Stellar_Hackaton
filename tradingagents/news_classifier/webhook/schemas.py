"""Pydantic schemas for the news classifier webhook."""

from pydantic import BaseModel, Field
from typing import Optional


class ClassifyRequest(BaseModel):
    title: str = Field(..., description="News article title")
    content: str = Field("", description="News article content")
    source: str = Field("", description="News source URL or name")
    url: str = Field("", description="Article URL")
    pub_date: str = Field("", description="Publication date")
    description: str = Field("", description="Article description/summary")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Classification: CRITICAL, MODERATE, or NORMAL")
    label_id: int = Field(..., description="Label ID: 0=NORMAL, 1=MODERATE, 2=CRITICAL")
    confidence: float = Field(..., description="Confidence score 0-1")
    probabilities: dict[str, float] = Field(..., description="Probabilities per class")
    title: str
    description: str = Field("", description="Article description/summary")
    url: str = Field("", description="Article URL")
    pub_date: str = Field("", description="Publication date")
    source: str = Field("", description="Clean source name")


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str = "0.2.0"
