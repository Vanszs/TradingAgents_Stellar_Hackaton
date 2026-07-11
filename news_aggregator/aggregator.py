"""News aggregator - polls RSS and classifies articles."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from news_aggregator.rss_monitor import poll_rss_feeds

logger = logging.getLogger(__name__)


class NewsAggregator:
    def __init__(self, config_path: str = "news_aggregator/config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.classifier = None
        self.seen_count = 0
        self.classified_count = 0

    def load_classifier(self):
        from tradingagents.news_classifier.inference.classifier import NewsClassifier
        self.classifier = NewsClassifier()
        logger.info("Classifier loaded")

    def classify_articles(self, articles: list[dict]) -> dict:
        results = {"NORMAL": [], "MODERATE": [], "CRITICAL": []}

        for article in articles:
            result = self.classifier.classify(
                title=article["title"],
                content=article.get("description", ""),
                source=article.get("source", ""),
                url=article.get("link", ""),
                pub_date=article.get("pub_date", ""),
                description=article.get("description", ""),
            )
            article["classification"] = result
            results[result["label"]].append(article)

        return results

    def log_results(self, results: dict, new_count: int):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info("[%s] NEW ARTICLES: %d", now, new_count)
        logger.info("[%s] NORMAL: %d articles", now, len(results["NORMAL"]))
        logger.info("[%s] MODERATE: %d articles", now, len(results["MODERATE"]))
        logger.info("[%s] CRITICAL: %d articles", now, len(results["CRITICAL"]))

        for article in results["CRITICAL"]:
            confidence = article["classification"]["confidence"]
            logger.info("  [CRITICAL] \"%s\" (confidence: %.2f)", article["title"][:60], confidence)

        for article in results["MODERATE"]:
            confidence = article["classification"]["confidence"]
            logger.info("  [MODERATE] \"%s\" (confidence: %.2f)", article["title"][:60], confidence)

    def save_fallback(self, results: dict) -> Path:
        fallback = results["CRITICAL"] + results["MODERATE"]
        if not fallback:
            return None

        output_path = Path("news_aggregator/data/fallback_articles.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", encoding="utf-8") as f:
            for article in fallback:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        logger.info("Saved %d fallback articles to %s", len(fallback), output_path)
        return output_path

    async def run(self):
        self.load_classifier()
        interval = self.config.get("interval_seconds", 900)

        logger.info("Starting News Aggregator (interval: %ds)", interval)

        while True:
            try:
                articles = poll_rss_feeds(self.config["rss_feeds"])

                if articles:
                    results = self.classify_articles(articles)
                    self.log_results(results, len(articles))
                    self.save_fallback(results)
                    self.classified_count += len(articles)
                else:
                    logger.info("No new articles found")

                logger.info("Next poll in %ds (total classified: %d)", interval, self.classified_count)

            except Exception as e:
                logger.error("Error in aggregator loop: %s", e)

            await asyncio.sleep(interval)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    aggregator = NewsAggregator()
    await aggregator.run()


if __name__ == "__main__":
    asyncio.run(main())
