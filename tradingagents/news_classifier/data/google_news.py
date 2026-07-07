"""Google News RSS fetcher for crypto news."""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
from xml.etree import ElementTree

from tradingagents.news_classifier.config import GOOGLE_NEWS_QUERIES, GOOGLE_NEWS_RSS_URL

logger = logging.getLogger(__name__)

USER_AGENT = "TradingAgents/1.0 (Crypto News Classifier)"


def _fetch_google_news_query(query: str, timeout: int = 30) -> list[dict]:
    articles = []
    url = GOOGLE_NEWS_RSS_URL.format(query=query.replace(" ", "+"))

    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            tree = ElementTree.parse(resp)

        root = tree.getroot()

        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            source_el = item.find("source")
            source_name = source_el.text.strip() if source_el is not None else "Google News"
            pub_date = item.findtext("pubDate", "").strip()

            if not title:
                continue

            article_id = hashlib.md5(f"gn_{title}{link}".encode()).hexdigest()
            articles.append({
                "id": article_id,
                "title": title,
                "link": link,
                "description": "",
                "source": f"google_news:{source_name}",
                "pub_date": pub_date,
                "query": query,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })

        logger.info("Google News query '%s': %d articles", query, len(articles))

    except (URLError, ElementTree.ParseError) as e:
        logger.warning("Failed to fetch Google News for '%s': %s", query, e)

    return articles


def collect_google_news(
    queries: Optional[list[str]] = None,
    deduplicate: bool = True,
) -> list[dict]:
    queries = queries or GOOGLE_NEWS_QUERIES
    all_articles = []
    seen_ids = set()

    for query in queries:
        articles = _fetch_google_news_query(query)
        for article in articles:
            if deduplicate and article["id"] in seen_ids:
                continue
            seen_ids.add(article["id"])
            all_articles.append(article)

    logger.info("Google News: %d unique articles from %d queries", len(all_articles), len(queries))
    return all_articles
