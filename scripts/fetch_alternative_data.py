#!/usr/bin/env python3
"""
Fetch news, fundamentals, and sentiment for Indonesian stocks from
alternative sources when Yahoo Finance returns empty data.

Usage:
    python scripts/fetch_alternative_data.py DEWA.JK BREN.JK TPIA.JK
    python scripts/fetch_alternative_data.py --all
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}


# --------------------------------------------------------------------------
# News fetching
# --------------------------------------------------------------------------

def fetch_news_google(ticker: str, days: int = 7) -> list[dict]:
    """Fetch news from Google News for Indonesian stocks.

    Searches for "{ticker} saham" in Indonesian news.
    Returns list of dicts with published_at, title, summary, source.
    """
    # Remove .JK suffix for search
    symbol = ticker.replace(".JK", "").replace(".jk", "")
    query = f"{symbol} saham indonesia"
    url = f"https://news.google.com/search?q={requests.utils.quote(query)}&hl=id&gl=ID&ceid=ID:id"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print(f"  [WARN] Google News fetch failed for {ticker}: {exc}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    # Google News uses various article selectors
    for item in soup.select("article, div.SoaJf, div.xrnccd"):
        title_elem = item.select_one("a.JtKRv, a.DY5T1d, h3 a, h4 a")
        if not title_elem:
            continue

        title = title_elem.get_text(strip=True)
        link = title_elem.get("href", "")

        # Make relative URLs absolute
        if link and not link.startswith("http"):
            link = f"https://news.google.com{link}"

        # Extract source
        source_elem = item.select_one(
            "span.vr1PY, span.wEwyrc, div.QmrM6d, span[data-n-tid]"
        )
        source = source_elem.get_text(strip=True) if source_elem else "Google News"

        # Extract time
        time_elem = item.select_one("time")
        pub_time = ""
        if time_elem:
            pub_time = time_elem.get("datetime", "")
            if not pub_time:
                pub_time = time_elem.get_text(strip=True)

        # Extract summary if available
        summary_elem = item.select_one(
            "span.St4GI, div.GI74Re, div.s3v9rd, p"
        )
        summary = summary_elem.get_text(strip=True) if summary_elem else ""

        articles.append({
            "published_at": pub_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "summary": summary,
            "source": source,
            "url": link,
        })

    return articles[:50]  # Cap at 50 articles


def fetch_news_detik(symbol: str, days: int = 7) -> list[dict]:
    """Fetch news from Detik Finance for Indonesian stocks."""
    query = symbol.replace(".JK", "").replace(".jk", "")
    url = f"https://www.detik.com/search/searchall?query={requests.utils.quote(query)}+saham&sitefinance"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print(f"  [WARN] Detik fetch failed for {symbol}: {exc}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    for item in soup.select("article, div.list-content__item, div.media__text"):
        title_elem = item.select_one("h2 a, h3 a, a.media__link")
        if not title_elem:
            continue

        title = title_elem.get_text(strip=True)
        link = title_elem.get("href", "")

        time_elem = item.select_one("span.date, time, span[class*='date']")
        pub_time = time_elem.get_text(strip=True) if time_elem else ""

        summary_elem = item.select_one("p, div.media__desc, span[class*='desc']")
        summary = summary_elem.get_text(strip=True) if summary_elem else ""

        articles.append({
            "published_at": pub_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "summary": summary,
            "source": "Detik",
            "url": link,
        })

    return articles[:30]


def fetch_news_kontan(symbol: str, days: int = 7) -> list[dict]:
    """Fetch news from Kontan for Indonesian stocks."""
    query = symbol.replace(".JK", "").replace(".jk", "")
    url = f"https://www.kontan.co.id/search/?search={requests.utils.quote(query)}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print(f"  [WARN] Kontan fetch failed for {symbol}: {exc}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    for item in soup.select("article, div.item, div.list-berita__item"):
        title_elem = item.select_one("h1 a, h2 a, h3 a, a.title")
        if not title_elem:
            continue

        title = title_elem.get_text(strip=True)
        link = title_elem.get("href", "")
        if link and not link.startswith("http"):
            link = f"https://www.kontan.co.id{link}"

        time_elem = item.select_one("time, span.date, span[class*='time']")
        pub_time = ""
        if time_elem:
            pub_time = time_elem.get("datetime", time_elem.get_text(strip=True))

        summary_elem = item.select_one("p, div.desc, span[class*='desc']")
        summary = summary_elem.get_text(strip=True) if summary_elem else ""

        articles.append({
            "published_at": pub_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "summary": summary,
            "source": "Kontan",
            "url": link,
        })

    return articles[:30]


def fetch_news_combined(ticker: str) -> list[dict]:
    """Fetch news from multiple sources and deduplicate."""
    all_news = []

    # Source 1: Google News
    print(f"    Fetching from Google News...")
    google_news = fetch_news_google(ticker)
    print(f"    → {len(google_news)} articles")
    all_news.extend(google_news)

    time.sleep(1)  # Rate limiting

    # Source 2: Detik
    print(f"    Fetching from Detik...")
    detik_news = fetch_news_detik(ticker)
    print(f"    → {len(detik_news)} articles")
    all_news.extend(detik_news)

    time.sleep(1)

    # Source 3: Kontan
    print(f"    Fetching from Kontan...")
    kontan_news = fetch_news_kontan(ticker)
    print(f"    → {len(kontan_news)} articles")
    all_news.extend(kontan_news)

    # Deduplicate by title similarity
    seen_titles = set()
    unique_news = []
    for article in all_news:
        title_key = article["title"][:50].lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_news.append(article)

    # Sort by published_at (most recent first)
    unique_news.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    return unique_news


# --------------------------------------------------------------------------
# Fundamentals fetching
# --------------------------------------------------------------------------

def fetch_fundamentals_yfinance(ticker: str) -> list[dict]:
    """Fetch fundamentals from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print(f"  [WARN] yfinance not installed, skipping fundamentals")
        return []

    print(f"    Fetching from Yahoo Finance...")
    stock = yf.Ticker(ticker)
    records = []

    # Company info
    try:
        info = stock.info or {}
        skip_keys = {
            "companyOfficers", "address1", "address2", "city", "state",
            "zip", "country", "phone", "website", "logo_url",
        }
        for key, value in info.items():
            if value is not None and key not in skip_keys:
                records.append({
                    "metric": key,
                    "value": value,
                    "available_date": datetime.now().strftime("%Y-%m-%d"),
                    "period": "latest",
                    "source": "yfinance_info",
                })
    except Exception as exc:
        print(f"    [WARN] Failed to fetch info: {exc}")

    # Quarterly financial statements
    for stmt_name, stmt_attr in [
        ("balance_sheet", "quarterly_balance_sheet"),
        ("income_statement", "quarterly_income_stmt"),
        ("cashflow", "quarterly_cashflow"),
    ]:
        try:
            stmt = getattr(stock, stmt_attr, None)
            if stmt is not None and not stmt.empty:
                for col in stmt.columns:
                    date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
                    for idx in stmt.index:
                        val = stmt.loc[idx, col]
                        if pd.notna(val):
                            records.append({
                                "metric": f"{stmt_name}:{idx}",
                                "value": float(val) if isinstance(val, (int, float)) else str(val),
                                "available_date": date_str,
                                "period": "quarterly",
                                "source": f"yfinance_{stmt_attr}",
                            })
        except Exception as exc:
            print(f"    [WARN] Failed to fetch {stmt_name}: {exc}")

    print(f"    → {len(records)} fundamental records")
    return records


# --------------------------------------------------------------------------
# Sentiment generation
# --------------------------------------------------------------------------

POSITIVE_WORDS = {
    "naik", "untung", "profit", "bullish", "positif", "bagus", "cuan",
    "gacor", "rocket", "moon", "buy", "accumulate", "outperform",
    "upgrade", "beat", "exceed", "strong", "growth", "rebound",
    "recovery", "breakout", "rally", "surge", "jump", "gain",
}

NEGATIVE_WORDS = {
    "turun", "rugi", "loss", "bearish", "negatif", "jelek", "zonk",
    "crash", "dump", "sell", "reduce", "underperform", "downgrade",
    "miss", "weak", "decline", "drop", "fall", "plunge", "slump",
    "collapse", "bankruptcy", "default", "fraud", "investigation",
}


def generate_sentiment_from_news(news: list[dict]) -> list[dict]:
    """Generate synthetic sentiment from news headlines using keyword matching."""
    sentiment = []

    for article in news:
        title = article.get("title", "").lower()
        summary = article.get("summary", "").lower()
        text = f"{title} {summary}"

        pos_count = sum(1 for w in POSITIVE_WORDS if w in text)
        neg_count = sum(1 for w in NEGATIVE_WORDS if w in text)

        if pos_count > neg_count:
            score = min(0.5 + (pos_count - neg_count) * 0.1, 1.0)
            label = "Bullish"
        elif neg_count > pos_count:
            score = max(0.5 - (neg_count - pos_count) * 0.1, 0.0)
            label = "Bearish"
        else:
            score = 0.5
            label = "Neutral"

        sentiment.append({
            "timestamp": article.get("published_at", ""),
            "source": f"news_derived_{article.get('source', 'unknown')}",
            "score": round(score, 2),
            "label": label,
            "text": article.get("title", "")[:200],
        })

    return sentiment


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def fetch_and_save(ticker: str) -> None:
    """Fetch all data for a ticker and save to data/TICKER/."""
    print(f"\n{'='*60}")
    print(f"  Fetching data for {ticker}")
    print(f"{'='*60}")

    data_dir = Path("data") / ticker
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch news
    print(f"\n  [1/3] News:")
    news = fetch_news_combined(ticker)
    print(f"  Total: {len(news)} unique articles")
    with open(data_dir / "news.json", "w", encoding="utf-8") as f:
        json.dump(news, f, indent=2, ensure_ascii=False)

    # 2. Fetch fundamentals (from cache or yfinance)
    print(f"\n  [2/3] Fundamentals:")
    cache_dir = Path("api_cache") / ticker
    if (cache_dir / "fundamentals.json").exists():
        print(f"    Found in api_cache, copying...")
        import shutil
        shutil.copy(cache_dir / "fundamentals.json", data_dir / "fundamentals.json")
    else:
        fundamentals = fetch_fundamentals_yfinance(ticker)
        with open(data_dir / "fundamentals.json", "w", encoding="utf-8") as f:
            json.dump(fundamentals, f, indent=2, ensure_ascii=False)

    # 3. Generate sentiment from news
    print(f"\n  [3/3] Sentiment (derived from news):")
    sentiment = generate_sentiment_from_news(news)
    print(f"  Generated {len(sentiment)} sentiment records")
    with open(data_dir / "sentiment.json", "w", encoding="utf-8") as f:
        json.dump(sentiment, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n  Summary for {ticker}:")
    print(f"    News:        {len(news)} articles")
    print(f"    Fundamentals: {len(json.loads((data_dir / 'fundamentals.json').read_text()))} records")
    print(f"    Sentiment:   {len(sentiment)} records")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_alternative_data.py TICKER1 TICKER2 ...")
        print("       python scripts/fetch_alternative_data.py --all")
        print("")
        print("Examples:")
        print("  python scripts/fetch_alternative_data.py DEWA.JK")
        print("  python scripts/fetch_alternative_data.py DEWA.JK BREN.JK TPIA.JK")
        sys.exit(1)

    if "--all" in sys.argv:
        # Fetch all tickers in data/
        tickers = [d.name for d in Path("data").iterdir() if d.is_dir()]
    else:
        tickers = sys.argv[1:]

    print(f"Fetching data for: {', '.join(tickers)}")
    print(f"Data will be saved to: data/TICKER/")

    for ticker in tickers:
        try:
            fetch_and_save(ticker)
        except Exception as exc:
            print(f"\n  [ERROR] Failed for {ticker}: {exc}")

    print(f"\n{'='*60}")
    print(f"  Done! All data saved to data/ directory.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
