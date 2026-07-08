"""FastAPI webhook server for real-time crypto news classification."""

import json
import logging
from pathlib import Path
from typing import Optional

from tradingagents.news_classifier.webhook.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from tradingagents.news_classifier.inference.classifier import NewsClassifier
        _classifier = NewsClassifier()
    return _classifier


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Crypto News Classifier",
        description="Real-time crypto news importance classification",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        get_classifier()
        logger.info("Crypto News Classifier webhook started")

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model_loaded=_classifier is not None,
            version="0.2.0",
        )

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(request: ClassifyRequest):
        classifier = get_classifier()

        result = classifier.classify(
            title=request.title,
            content=request.content,
            source=request.source,
        )

        return ClassifyResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            title=result["title"],
            source=result["source"],
        )

    return app


def run_server(host: str = "0.0.0.0", port: int = 8001):
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
