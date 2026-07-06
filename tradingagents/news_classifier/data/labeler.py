"""LLM-assisted auto-labeling for crypto news articles."""

import json
import logging
from pathlib import Path
from typing import Optional

from tradingagents.news_classifier.config import LABEL_MAP, LABELER_PROMPT, DATA_DIR

logger = logging.getLogger(__name__)


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


def label_with_llm(
    articles: list[dict],
    llm_client=None,
    batch_size: int = 10,
) -> list[dict]:
    labeled = []

    if llm_client is None:
        try:
            from openai import OpenAI
            llm_client = OpenAI()
        except Exception:
            logger.error("No LLM client available. Provide llm_client or set OPENAI_API_KEY.")
            return articles

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
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            label_text = response.choices[0].message.content
            label = _parse_label(label_text)

            if label:
                article["label"] = label
                article["label_source"] = "llm"
                article["label_confidence"] = "auto"
            else:
                logger.warning("Could not parse label from LLM response: %s", label_text)
                article["label"] = "BIASA"
                article["label_source"] = "llm_fallback"
                article["label_confidence"] = "low"

        except Exception as e:
            logger.error("LLM labeling failed for article %d: %s", i, e)
            article["label"] = "BIASA"
            article["label_source"] = "error_fallback"
            article["label_confidence"] = "none"

        labeled.append(article)

        if (i + 1) % batch_size == 0:
            logger.info("Labeled %d/%d articles", i + 1, len(articles))

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
