"""CryptoCompare News API client."""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode

from tradingagents.news_classifier.config import CRYPTO_COMPARE_API_KEY, CRYPTO_COMPARE_BASE_URL

logger = logging.getLogger(__name__)


def fetch_crypto_compare_news(
    categories: str = "BTC,ETH,Trading,Exchange",
    lang: str = "EN",
    limit: int = 100,
    api_key: str = None,
) -> list[dict]:
    api_key = api_key or CRYPTO_COMPARE_API_KEY
    articles = []

    params = {
        "lang": lang,
        "sortOrder": "latest",
    }
    if categories:
        params["categories"] = categories
    if api_key:
        params["api_key"] = api_key

    url = f"{CRYPTO_COMPARE_BASE_URL}?{urlencode(params)}"

    try:
        req = Request(url, headers={"User-Agent": "TradingAgents/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("Type") != 100:
            logger.warning("CryptoCompare returned type: %s", data.get("Type"))
            return articles

        for item in data.get("Data", []):
            title = item.get("title", "").strip()
            if not title:
                continue

            body = item.get("body", "")
            source = item.get("source", "")
            categories_list = item.get("categories", "").split("|")
            published_on = item.get("published_on", 0)

            article_id = f"cc_{item.get('id', '')}"
            pub_date = datetime.fromtimestamp(published_on, tz=timezone.utc).isoformat() if published_on else ""

            articles.append({
                "id": article_id,
                "title": title,
                "description": body[:2000] if body else "",
                "link": item.get("url", ""),
                "source": f"crypto_compare:{source}",
                "pub_date": pub_date,
                "categories": categories_list,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })

        logger.info("Fetched %d articles from CryptoCompare", len(articles))

    except (URLError, json.JSONDecodeError) as e:
        logger.error("CryptoCompare fetch failed: %s", e)

    return articles


def fetch_crypto_compare_historical(
    categories: str = "BTC,ETH,Trading,Exchange",
    lang: str = "EN",
    max_articles: int = 1000,
    api_key: str = None,
) -> list[dict]:
    all_articles = []
    seen_ids = set()
    api_key = api_key or CRYPTO_COMPARE_API_KEY

    params = {
        "lang": lang,
        "sortOrder": "latest",
        "extraParams": "TradingAgents",
    }
    if categories:
        params["categories"] = categories
    if api_key:
        params["api_key"] = api_key

    url = f"{CRYPTO_COMPARE_BASE_URL}?{urlencode(params)}"

    try:
        req = Request(url, headers={"User-Agent": "TradingAgents/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("Type") != 100:
            return all_articles

        for item in data.get("Data", []):
            title = item.get("title", "").strip()
            if not title:
                continue

            article_id = f"cc_{item.get('id', '')}"
            if article_id in seen_ids:
                continue
            seen_ids.add(article_id)

            body = item.get("body", "")
            source = item.get("source", "")
            published_on = item.get("published_on", 0)

            all_articles.append({
                "id": article_id,
                "title": title,
                "description": body[:2000] if body else "",
                "link": item.get("url", ""),
                "source": f"crypto_compare:{source}",
                "pub_date": datetime.fromtimestamp(published_on, tz=timezone.utc).isoformat() if published_on else "",
                "categories": item.get("categories", "").split("|"),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })

            if len(all_articles) >= max_articles:
                break

        logger.info("Fetched %d historical articles from CryptoCompare", len(all_articles))

    except (URLError, json.JSONDecodeError) as e:
        logger.error("CryptoCompare historical fetch failed: %s", e)

    return all_articles
