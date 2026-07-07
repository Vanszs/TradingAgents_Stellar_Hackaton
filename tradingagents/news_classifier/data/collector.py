"""RSS news collector for crypto news training data."""

import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
from xml.etree import ElementTree

from tradingagents.news_classifier.config import RSS_FEEDS, DATA_DIR, CRYPTO_COMPARE_API_KEY

logger = logging.getLogger(__name__)

USER_AGENT = "TradingAgents/1.0 (Crypto News Classifier)"


def _fetch_feed(url: str, timeout: int = 30) -> list[dict]:
    articles = []
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            tree = ElementTree.parse(resp)

        root = tree.getroot()
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            desc = item.findtext("description", "").strip()
            pub_date = item.findtext("pubDate", "").strip()

            if not title:
                continue

            article_id = hashlib.md5(f"{title}{link}".encode()).hexdigest()
            articles.append({
                "id": article_id,
                "title": title,
                "link": link,
                "description": desc,
                "pub_date": pub_date,
                "source": url,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })

        for entry in root.findall(".//atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).strip()
            link_el = entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            desc = entry.findtext("atom:summary", "", ns).strip()
            pub_date = entry.findtext("atom:published", "", ns).strip()

            if not title:
                continue

            article_id = hashlib.md5(f"{title}{link}".encode()).hexdigest()
            articles.append({
                "id": article_id,
                "title": title,
                "link": link,
                "description": desc,
                "pub_date": pub_date,
                "source": url,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })

    except (URLError, ElementTree.ParseError) as e:
        logger.warning("Failed to fetch %s: %s", url, e)

    return articles


def collect_from_feeds(
    feeds: Optional[list[str]] = None,
    deduplicate: bool = True,
) -> list[dict]:
    all_articles = []
    seen_ids = set()

    for url in (feeds or RSS_FEEDS):
        articles = _fetch_feed(url)
        for article in articles:
            if deduplicate and article["id"] in seen_ids:
                continue
            seen_ids.add(article["id"])
            all_articles.append(article)

    logger.info("Collected %d unique articles from %d RSS feeds", len(all_articles), len(feeds or RSS_FEEDS))
    return all_articles


def collect_from_crypto_compare(
    api_key: str = None,
    max_articles: int = 1000,
) -> list[dict]:
    api_key = api_key or CRYPTO_COMPARE_API_KEY
    if not api_key:
        logger.warning("No CryptoCompare API key, skipping")
        return []

    from tradingagents.news_classifier.data.crypto_compare import fetch_crypto_compare_news
    articles = fetch_crypto_compare_news(api_key=api_key, limit=max_articles)
    return articles


def collect_all(
    rss_feeds: Optional[list[str]] = None,
    include_crypto_compare: bool = True,
    max_articles: int = 1000,
) -> list[dict]:
    all_articles = []
    seen_ids = set()

    rss_articles = collect_from_feeds(rss_feeds)
    for article in rss_articles:
        if article["id"] not in seen_ids:
            seen_ids.add(article["id"])
            all_articles.append(article)

    if include_crypto_compare:
        cc_articles = collect_from_crypto_compare(max_articles=max_articles)
        for article in cc_articles:
            if article["id"] not in seen_ids:
                seen_ids.add(article["id"])
                all_articles.append(article)

    logger.info("Total collected: %d unique articles (RSS + CryptoCompare)", len(all_articles))
    return all_articles


def save_articles(articles: list[dict], output_path: Optional[Path] = None) -> Path:
    output_path = output_path or (DATA_DIR / "collected_articles.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    logger.info("Saved %d articles to %s", len(articles), output_path)
    return output_path


def collect_and_save(
    mode: str = "rss",
    feeds: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    interval_seconds: int = 900,
    max_iterations: int = 1,
    max_articles: int = 1000,
) -> None:
    for i in range(max_iterations):
        if mode == "rss":
            articles = collect_from_feeds(feeds)
        elif mode == "all":
            articles = collect_all(feeds, include_crypto_compare=True, max_articles=max_articles)
        elif mode == "historical":
            articles = collect_from_crypto_compare(max_articles=max_articles)
        else:
            articles = collect_from_feeds(feeds)

        if articles:
            save_articles(articles, output_path)

        if i < max_iterations - 1:
            logger.info("Sleeping %d seconds until next collection...", interval_seconds)
            time.sleep(interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_and_save(mode="rss", max_iterations=1)
