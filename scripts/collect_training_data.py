"""Script to collect and label crypto news training data."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.news_classifier.data.collector import collect_from_feeds, save_articles
from tradingagents.news_classifier.data.labeler import (
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
    logger.info("=== Crypto News Training Data Collection ===")

    logger.info("Step 1: Collecting articles from RSS feeds...")
    articles = collect_from_feeds()
    if not articles:
        logger.error("No articles collected. Check RSS feed URLs and network connectivity.")
        sys.exit(1)

    raw_path = save_articles(articles)
    logger.info("Saved %d raw articles to %s", len(articles), raw_path)

    logger.info("Step 2: Labeling articles with LLM...")
    labeled_articles = label_with_llm(articles)

    labeled_path = save_labeled_articles(labeled_articles)

    dist = get_label_distribution(labeled_articles)
    logger.info("=== Label Distribution ===")
    for label, count in dist.items():
        logger.info("  %s: %d (%.1f%%)", label, count, 100 * count / max(len(labeled_articles), 1))

    logger.info("=== Done! ===")
    logger.info("Raw articles: %s", raw_path)
    logger.info("Labeled articles: %s", labeled_path)
    logger.info("Next step: python scripts/train_model.py")


if __name__ == "__main__":
    main()
