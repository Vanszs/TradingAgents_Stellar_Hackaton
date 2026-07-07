"""Script to collect and label crypto news training data."""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.news_classifier.config import (
    load_providers_config,
    get_provider_config,
    get_model_name,
    RSS_FEEDS,
    CRYPTO_COMPARE_API_KEY,
)
from tradingagents.news_classifier.data.collector import (
    collect_from_feeds,
    collect_from_crypto_compare,
    collect_all,
    save_articles,
)
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
    parser = argparse.ArgumentParser(description="Collect and label crypto news training data")
    parser.add_argument(
        "--mode",
        choices=["rss", "all", "historical"],
        default="rss",
        help="Collection mode: rss (RSS only), all (RSS + CryptoCompare), historical (CryptoCompare only)",
    )
    parser.add_argument("--max-articles", type=int, default=1000, help="Max articles to collect")
    parser.add_argument("--skip-labeling", action="store_true", help="Skip LLM labeling step")
    args = parser.parse_args()

    config = load_providers_config()
    provider_config = get_provider_config(config=config)
    model_name = get_model_name(provider_config)
    provider_label = provider_config.get("name", "unknown")

    logger.info("=== Crypto News Training Data Collection ===")
    logger.info("Mode: %s", args.mode)
    logger.info("Labeling Provider: %s", provider_label)
    logger.info("Labeling Model: %s", model_name)
    logger.info("RSS Feeds: %d sources", len(RSS_FEEDS))
    logger.info("CryptoCompare API Key: %s", "configured" if CRYPTO_COMPARE_API_KEY else "not set")

    if args.mode == "rss":
        logger.info("Step 1: Collecting from RSS feeds...")
        articles = collect_from_feeds()
    elif args.mode == "all":
        logger.info("Step 1: Collecting from RSS feeds + CryptoCompare...")
        articles = collect_all(include_crypto_compare=True, max_articles=args.max_articles)
    elif args.mode == "historical":
        if not CRYPTO_COMPARE_API_KEY:
            logger.error("CryptoCompare API key required for historical mode. Set CRYPTO_COMPARE_API_KEY in .env")
            sys.exit(1)
        logger.info("Step 1: Collecting historical data from CryptoCompare...")
        articles = collect_from_crypto_compare(max_articles=args.max_articles)
    else:
        articles = collect_from_feeds()

    if not articles:
        logger.error("No articles collected. Check network connectivity and API keys.")
        sys.exit(1)

    raw_path = save_articles(articles)
    logger.info("Saved %d raw articles to %s", len(articles), raw_path)

    if args.skip_labeling:
        logger.info("Skipping labeling (--skip-labeling flag)")
        logger.info("Next step: python scripts/train_model.py")
        return

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
