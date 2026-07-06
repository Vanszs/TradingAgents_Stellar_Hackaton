"""Configuration for the crypto news classifier."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
DATA_DIR = PROJECT_ROOT / "data" / "cache"

LABEL_MAP = {"BIASA": 0, "LUMAYAN": 1, "PENTING": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

MODEL_NAME = os.getenv("NEWS_CLASSIFIER_MODEL", "bert-base-uncased")
MAX_LENGTH = int(os.getenv("NEWS_CLASSIFIER_MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("NEWS_CLASSIFIER_BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("NEWS_CLASSIFIER_LR", "2e-5"))
NUM_EPOCHS = int(os.getenv("NEWS_CLASSIFIER_EPOCHS", "5"))
WARMUP_RATIO = float(os.getenv("NEWS_CLASSIFIER_WARMUP", "0.1"))

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss.xml",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://cryptonews.com/news/feed/",
]

CRYPTO_KEYWORDS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "AVAX",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "ETC", "FIL", "APT", "ARB",
    "OP", "NEAR", "ICP", "FTM", "ALGO", "XLM", "VET", "HBAR", "MANA",
    "SAND", "AXS", "GALA", "BITCOIN", "ETHEREUM", "SOLANA", "BINANCE",
    "RIPPLE", "CARDANO", "DOGECOIN", "POLKADOT", "AVALANCHE", "POLYGON",
    "CHAINLINK", "UNISWAP", "COSMOS", "LITECOIN",
}

SANITIZER_PROMPT = """You are a crypto news impact assessor. A machine learning model classified this news as "PENTING" (critical).

Your job is to VALIDATE this classification by assessing:
1. Is this genuinely impactful news for crypto markets?
2. What specific coins/tokens are affected?
3. Is the narrative positive, negative, or mixed?
4. What is the expected market impact?

Respond in JSON format:
{
  "validated": true/false,
  "affected_coins": ["BTC", "ETH", ...],
  "narrative_type": "positive"|"negative"|"mixed",
  "impact_summary": "brief description",
  "confidence": 0.0-1.0
}

News to assess:
Title: {title}
Content: {content}
Source: {source}
"""

LABELER_PROMPT = """You are a crypto news classifier. Classify the following news article into one of 3 levels:

- PENTING: Highly impactful news that could significantly move crypto markets. Examples: major exchange hack/collapse, ETF approval/rejection, major regulatory action, huge whale movements, protocol critical vulnerability, major partnership listing.
- LUMAYAN: Moderately impactful news. Examples: minor exchange issues, project updates, moderate regulatory news, market analysis with notable insights, minor hack/exploit.
- BIASA: Regular/routine news. Examples: price updates, routine project announcements, opinion pieces, minor market commentary.

Respond with ONLY the classification label: PENTING, LUMAYAN, or BIASA.

Title: {title}
Content: {content}
Source: {source}
"""
