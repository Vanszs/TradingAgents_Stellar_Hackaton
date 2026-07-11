"""Script to label collected articles (no re-collection)."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.news_classifier.config import (
    load_providers_config,
    get_provider_config,
    get_model_name,
)
from tradingagents.news_classifier.data.collector import save_articles
from tradingagents.news_classifier.data.labeler import (
    load_unlabeled_articles,
    label_with_llm,
    save_labeled_articles,
    get_label_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_providers_config()
    provider_config = get_provider_config(config=config)
    model_name = get_model_name(provider_config)

    logger.info("=== Labeling Collected Articles ===")
    logger.info("Provider: %s", provider_config.get("name"))
    logger.info("Model: %s", model_name)

    articles = load_unlabeled_articles()
    if not articles:
        logger.error("No collected articles found.")
        sys.exit(1)

    logger.info("Found %d articles to label", len(articles))

    labeled_articles = label_with_llm(articles)

    labeled_path = save_labeled_articles(labeled_articles)

    dist = get_label_distribution(labeled_articles)
    logger.info("=== Label Distribution ===")
    for label, count in dist.items():
        logger.info("  %s: %d (%.1f%%)", label, count, 100 * count / max(len(labeled_articles), 1))

    logger.info("=== Done! ===")
    logger.info("Labeled articles: %s", labeled_path)


if __name__ == "__main__":
    main()
