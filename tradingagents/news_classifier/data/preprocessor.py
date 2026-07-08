"""Text preprocessing for crypto news articles."""

import re
import html
from typing import Optional


def clean_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_crypto_terms(text: str) -> str:
    replacements = {
        r"\bBTC\b": "Bitcoin",
        r"\bETH\b": "Ethereum",
        r"\bSOL\b": "Solana",
        r"\bBNB\b": "Binance Coin",
        r"\bXRP\b": "Ripple",
        r"\bADA\b": "Cardano",
        r"\bDOGE\b": "Dogecoin",
        r"\bDOT\b": "Polkadot",
        r"\bAVAX\b": "Avalanche",
        r"\bMATIC\b": "Polygon",
        r"\bLINK\b": "Chainlink",
        r"\bUNI\b": "Uniswap",
        r"\bDeFi\b": "Decentralized Finance",
        r"\bNFT\b": "Non-Fungible Token",
        r"\bDAO\b": "Decentralized Autonomous Organization",
        r"\bDEX\b": "Decentralized Exchange",
        r"\bCEX\b": "Centralized Exchange",
        r"\bTVL\b": "Total Value Locked",
        r"\bAPY\b": "Annual Percentage Yield",
        r"\bHODL\b": "Hold",
        r"\bFUD\b": "Fear Uncertainty Doubt",
        r"\bFOMO\b": "Fear Of Missing Out",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text)


def remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+\.\S+", "", text)


def remove_special_chars(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_length: int = 512) -> str:
    words = text.split()
    if len(words) > max_length:
        return " ".join(words[:max_length])
    return text


def preprocess_article(
    title: Optional[str],
    content: Optional[str],
    normalize_crypto: bool = True,
    max_words: int = 400,
) -> str:
    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    if content:
        parts.append(f"CONTENT: {content}")

    text = " ".join(parts)
    text = clean_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = normalize_whitespace(text)

    if normalize_crypto:
        text = normalize_crypto_terms(text)

    text = remove_special_chars(text)
    text = truncate_text(text, max_words)
    return text


# ===== FEATURE ENGINEERING =====

HACK_KEYWORDS = [
    "hack", "hacked", "exploit", "exploited", "stolen", "breach",
    "compromised", "vulnerability", "attack", "drained", "theft",
    "rug pull", "scam", "fraud", "ponzi",
]

REGULATION_KEYWORDS = [
    "sec", "regulation", "regulatory", "enforcement", "ban",
    "approval", "rejection", "lawsuit", "legal", "compliance",
    "mica", "clarity act", "etf",
]

WHALE_KEYWORDS = [
    "whale", "million", "billion", "transfer", "move",
    "dump", "sell", "buy", "acquisition",
]

MARKET_KEYWORDS = [
    "crash", "crash", "plunge", "surge", "rally",
    "bull", "bear", "correction", "volatile", "liquidation",
]

EXCHANGE_KEYWORDS = [
    "exchange", "binance", "coinbase", "kraken", "ftx",
    "delisting", "freeze", "withdrawal", "insolvency", "bankruptcy",
]


def extract_features(article: dict) -> dict:
    title = article.get("title", "").lower()
    desc = article.get("description", "").lower()
    text = f"{title} {desc}"

    features = {}

    features["has_dollar_amount"] = bool(re.search(r"\$[\d,.]+[mbk]?", text))
    features["has_hack_keyword"] = any(kw in text for kw in HACK_KEYWORDS)
    features["has_regulation_keyword"] = any(kw in text for kw in REGULATION_KEYWORDS)
    features["has_whale_keyword"] = any(kw in text for kw in WHALE_KEYWORDS)
    features["has_market_keyword"] = any(kw in text for kw in MARKET_KEYWORDS)
    features["has_exchange_keyword"] = any(kw in text for kw in EXCHANGE_KEYWORDS)
    features["title_length"] = len(article.get("title", "").split())
    features["desc_length"] = len(article.get("description", "").split())
    features["total_length"] = features["title_length"] + features["desc_length"]

    source = article.get("source", "")
    if "coindesk" in source.lower():
        features["source_coindesk"] = 1
    elif "cointelegraph" in source.lower():
        features["source_cointelegraph"] = 1
    elif "decrypt" in source.lower():
        features["source_decrypt"] = 1
    elif "google_news" in source.lower():
        features["source_google"] = 1
    else:
        features["source_other"] = 1

    return features


def preprocess_with_features(article: dict, normalize_crypto: bool = True, max_words: int = 400) -> str:
    title = article.get("title", "")
    desc = article.get("description", "")

    features = extract_features(article)

    feature_tags = []
    if features.get("has_hack_keyword"):
        feature_tags.append("[SECURITY]")
    if features.get("has_regulation_keyword"):
        feature_tags.append("[REGULATION]")
    if features.get("has_whale_keyword"):
        feature_tags.append("[WHALE]")
    if features.get("has_market_keyword"):
        feature_tags.append("[MARKET]")
    if features.get("has_exchange_keyword"):
        feature_tags.append("[EXCHANGE]")
    if features.get("has_dollar_amount"):
        feature_tags.append("[FINANCIAL]")

    parts = []
    if feature_tags:
        parts.append(" ".join(feature_tags))
    if title:
        parts.append(f"TITLE: {title}")
    if desc:
        parts.append(f"CONTENT: {desc}")

    text = " ".join(parts)
    text = clean_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = normalize_whitespace(text)

    if normalize_crypto:
        text = normalize_crypto_terms(text)

    text = remove_special_chars(text)
    text = truncate_text(text, max_words)
    return text
