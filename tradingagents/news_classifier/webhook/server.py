"""FastAPI webhook server for real-time crypto news classification."""

import json
import logging
from pathlib import Path
from typing import Optional

from tradingagents.news_classifier.webhook.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    HealthResponse,
    ActiveCoinsConfig,
)

logger = logging.getLogger(__name__)

_active_coins_config: Optional[ActiveCoinsConfig] = None
_classifier = None
_sanitizer = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from tradingagents.news_classifier.inference.classifier import NewsClassifier
        _classifier = NewsClassifier()
    return _classifier


def get_sanitizer():
    global _sanitizer
    if _sanitizer is None:
        from tradingagents.news_classifier.inference.sanitizer import NewsSanitizer
        _sanitizer = NewsSanitizer()
    return _sanitizer


def load_active_coins(config_path: Optional[Path] = None) -> ActiveCoinsConfig:
    global _active_coins_config
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "active_coins.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            data = json.load(f)
        _active_coins_config = ActiveCoinsConfig(**data)
        logger.info("Loaded %d active coins", len(_active_coins_config.coins))
    else:
        _active_coins_config = ActiveCoinsConfig(coins=[])
        logger.warning("No active_coins.json found at %s", config_path)

    return _active_coins_config


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Crypto News Classifier",
        description="Real-time crypto news importance classification with AI sanitization",
        version="0.1.0",
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
        get_sanitizer()
        load_active_coins()
        logger.info("Crypto News Classifier webhook started")

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model_loaded=_classifier is not None,
            version="0.1.0",
        )

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(request: ClassifyRequest):
        classifier = get_classifier()
        sanitizer = get_sanitizer()

        result = classifier.classify(
            title=request.title,
            content=request.content,
            source=request.source,
        )

        response = ClassifyResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            title=result["title"],
            source=result["source"],
        )

        if result["label"] == "PENTING":
            sanitized = sanitizer.sanitize(
                title=request.title,
                content=request.content,
                source=request.source,
                classifier_confidence=result["confidence"],
            )
            response.sanitized = sanitized

            if sanitized.get("validated", False):
                config = _active_coins_config or ActiveCoinsConfig(coins=[])
                active_tickers = [c.ticker for c in config.coins]
                matches = sanitizer.matches_active_coins(sanitized, active_tickers)
                response.triggered_coins = matches

                if matches:
                    logger.info("Triggered re-analysis for coins: %s", matches)
                    _trigger_analysis(matches, request.title, sanitized)

        return response

    @app.post("/classify/batch", response_model=BatchClassifyResponse)
    async def classify_batch(request: BatchClassifyRequest):
        classifier = get_classifier()
        results = []

        for article in request.articles:
            result = classifier.classify(
                title=article.title,
                content=article.content,
                source=article.source,
            )
            results.append(ClassifyResponse(
                label=result["label"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                title=result["title"],
                source=result["source"],
            ))

        return BatchClassifyResponse(results=results)

    @app.get("/active-coins")
    async def get_active_coins():
        config = _active_coins_config or ActiveCoinsConfig(coins=[])
        return config

    return app


def _trigger_analysis(coins: list[str], news_title: str, sanitized: dict):
    logger.info("Analysis trigger: coins=%s, news='%s'", coins, news_title[:80])
    logger.info("Impact: %s", sanitized.get("impact_summary", "N/A"))
    logger.info("Narrative: %s", sanitized.get("narrative_type", "N/A"))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
