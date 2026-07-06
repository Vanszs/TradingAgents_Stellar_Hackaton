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
