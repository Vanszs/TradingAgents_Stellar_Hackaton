"""Pydantic schemas for the news classifier webhook."""

from pydantic import BaseModel, Field
from typing import Optional


class ClassifyRequest(BaseModel):
    title: str = Field(..., description="News article title")
    content: str = Field("", description="News article content")
    source: str = Field("", description="News source URL or name")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Classification: PENTING, LUMAYAN, or BIASA")
    confidence: float = Field(..., description="Confidence score 0-1")
    probabilities: dict[str, float] = Field(..., description="Probabilities per class")
    title: str
    source: str
    sanitized: Optional[dict] = Field(None, description="Sanitization result for PENTING")
    triggered_coins: Optional[list[str]] = Field(None, description="Active coins that matched")


class BatchClassifyRequest(BaseModel):
    articles: list[ClassifyRequest]


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class WebhookPayload(BaseModel):
    event: str = Field("news.classified", description="Event type")
    data: ClassifyResponse


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str = "0.1.0"


class ActiveCoin(BaseModel):
    ticker: str
    name: str
    narratives: list[str] = Field(default_factory=list, description="Key narratives to watch")


class ActiveCoinsConfig(BaseModel):
    coins: list[ActiveCoin] = Field(default_factory=list)
