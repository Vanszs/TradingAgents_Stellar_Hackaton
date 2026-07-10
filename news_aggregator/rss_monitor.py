"""RSS feed monitor for news aggregation."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

USER_AGENT = "TradingAgents/1.0 (News Aggregator)"
SEEN_FILE = Path("news_aggregator/seen_articles.json")


def load_seen_ids() -> set:
    if SEEN_FILE.exists():
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids: set):
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen_ids), f)


def poll_rss_feeds(feeds: list[str]) -> list[dict]:
    seen_ids = load_seen_ids()
    new_articles = []

    for url in feeds:
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                tree = ElementTree.parse(resp)

            root = tree.getroot()
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                desc = item.findtext("description", "").strip()
                pub_date = item.findtext("pubDate", "").strip()

                if not title:
                    continue

                article_id = hashlib.md5(f"{title}{link}".encode()).hexdigest()

                if article_id in seen_ids:
                    continue

                seen_ids.add(article_id)
                new_articles.append({
                    "id": article_id,
                    "title": title,
                    "description": desc,
                    "link": link,
                    "source": url,
                    "pub_date": pub_date,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                })

        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)

    save_seen_ids(seen_ids)
    return new_articles
