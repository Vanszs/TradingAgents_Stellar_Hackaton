"""LLM-assisted auto-labeling for crypto news articles."""

import json
import logging
from pathlib import Path
from typing import Optional

from tradingagents.news_classifier.config import (
    LABEL_MAP,
    LABELER_PROMPT,
    DATA_DIR,
    load_providers_config,
    get_provider_config,
    get_model_name,
    get_api_key,
    get_base_url,
    get_labeling_config,
)

logger = logging.getLogger(__name__)


def create_llm_client(
    provider_name: str = None,
    config: dict = None,
):
    config = config or load_providers_config()
    provider_config = get_provider_config(provider_name, config)
    api_key = get_api_key(provider_config)
    base_url = get_base_url(provider_config)
    provider_label = provider_config.get("name", provider_name or "unknown")

    if not api_key:
        logger.error(
            "No API key found for provider '%s'. Set env var '%s'.",
            provider_label,
            provider_config.get("api_key_env", "API_KEY"),
        )
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        logger.info("Initialized LLM client: provider=%s, base_url=%s", provider_label, base_url)
        return client
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return None
    except Exception as e:
        logger.error("Failed to create LLM client for '%s': %s", provider_label, e)
        return None


def _build_label_prompt(title: str, content: str, source: str) -> str:
    return LABELER_PROMPT.format(
        title=title,
        content=content[:2000],
        source=source,
    )


def _parse_label(response: str) -> Optional[str]:
    response = response.strip().upper()
    for label in LABEL_MAP:
        if label in response:
            return label
    return None


def _save_checkpoint(articles: list[dict]) -> None:
    checkpoint_path = DATA_DIR / "labeled_checkpoint.jsonl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")


def label_with_llm(
    articles: list[dict],
    llm_client=None,
    provider_name: str = None,
    model_name: str = None,
    batch_size: int = None,
) -> list[dict]:
    config = load_providers_config()
    labeling_config = get_labeling_config(config)

    batch_size = batch_size or labeling_config.get("batch_size", 10)
    temperature = labeling_config.get("temperature", 0.0)
    max_tokens = labeling_config.get("max_tokens", 10)
    save_interval = 20

    if llm_client is None:
        llm_client = create_llm_client(provider_name, config)
        if llm_client is None:
            logger.error("Cannot create LLM client. Returning unlabeled articles.")
            return articles

    if model_name is None:
        provider_config = get_provider_config(provider_name, config)
        model_name = get_model_name(provider_config)

    logger.info("Labeling %d articles with model=%s", len(articles), model_name)

    labeled = []
    success_count = 0
    error_count = 0

    for i, article in enumerate(articles):
        if "label" in article:
            labeled.append(article)
            continue

        prompt = _build_label_prompt(
            article.get("title", ""),
            article.get("description", ""),
            article.get("source", ""),
        )

        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            label_text = response.choices[0].message.content
            label = _parse_label(label_text)

            if label:
                article["label"] = label
                article["label_source"] = f"llm:{model_name}"
                article["label_confidence"] = "auto"
                success_count += 1
            else:
                logger.warning("Could not parse label from response: %s", label_text)
                article["label"] = "NORMAL"
                article["label_source"] = "llm_fallback"
                article["label_confidence"] = "low"
                error_count += 1

        except Exception as e:
            logger.error("LLM labeling failed for article %d: %s", i, e)
            article["label"] = "NORMAL"
            article["label_source"] = "error_fallback"
            article["label_confidence"] = "none"
            error_count += 1

        labeled.append(article)

        if (i + 1) % batch_size == 0:
            logger.info(
                "Progress: %d/%d (success=%d, errors=%d)",
                i + 1, len(articles), success_count, error_count,
            )

        if (i + 1) % save_interval == 0:
            _save_checkpoint(labeled)

    _save_checkpoint(labeled)

    logger.info(
        "Labeling complete: %d articles (success=%d, errors=%d)",
        len(labeled), success_count, error_count,
    )
    return labeled


def save_labeled_articles(articles: list[dict], output_path: Optional[Path] = None) -> Path:
    output_path = output_path or (DATA_DIR / "labeled_articles.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    logger.info("Saved %d labeled articles to %s", len(articles), output_path)
    return output_path


def load_unlabeled_articles(input_path: Optional[Path] = None) -> list[dict]:
    input_path = input_path or (DATA_DIR / "collected_articles.jsonl")
    if not input_path.exists():
        logger.warning("No articles file found at %s", input_path)
        return []

    articles = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))

    logger.info("Loaded %d articles from %s", len(articles), input_path)
    return articles


def get_label_distribution(articles: list[dict]) -> dict[str, int]:
    dist = {"PENTING": 0, "LUMAYAN": 0, "BIASA": 0, "UNLABELED": 0}
    for article in articles:
        label = article.get("label", "UNLABELED")
        dist[label] = dist.get(label, 0) + 1
    return dist
