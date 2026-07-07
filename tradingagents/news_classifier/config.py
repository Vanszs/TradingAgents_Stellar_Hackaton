"""Configuration for the crypto news classifier."""

import json
import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
DATA_DIR = PROJECT_ROOT / "data" / "cache"
PROVIDERS_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "llm_providers.json"

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
    "https://beincrypto.com/feed/",
    "https://u.today/rss",
    "https://cryptoslate.com/feed/",
    "https://bravenewcoin.com/feed",
]

CRYPTO_COMPARE_API_KEY = os.getenv("CRYPTO_COMPARE_API_KEY", "")
CRYPTO_COMPARE_BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"

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


def load_providers_config(config_path: Path = None) -> dict:
    config_path = config_path or PROVIDERS_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Providers config not found at %s, using defaults", config_path)
        return _default_providers_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to load providers config: %s", e)
        return _default_providers_config()


def _default_providers_config() -> dict:
    return {
        "providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "default_model": "gpt-4o-mini",
                "models": ["gpt-4o-mini", "gpt-4o"],
            }
        },
        "default_provider": "openai",
        "default_model": "gpt-4o-mini",
        "labeling": {
            "temperature": 0.0,
            "max_tokens": 10,
            "batch_size": 10,
        },
    }


def get_provider_config(provider_name: str = None, config: dict = None) -> dict:
    config = config or load_providers_config()
    provider_name = provider_name or os.getenv("NEWS_LABELER_PROVIDER") or config.get("default_provider", "openai")

    providers = config.get("providers", {})
    if provider_name not in providers:
        logger.warning("Provider '%s' not found, available: %s", provider_name, list(providers.keys()))
        provider_name = config.get("default_provider", "openai")

    provider = providers.get(provider_name, {})
    provider["name"] = provider_name
    return provider


def get_model_name(provider_config: dict) -> str:
    return os.getenv("NEWS_LABELER_MODEL") or provider_config.get("default_model", "gpt-4o-mini")


def get_api_key(provider_config: dict) -> str:
    api_key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
    return os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY") or os.getenv("NEWS_LABELER_API_KEY", "")


def get_base_url(provider_config: dict) -> str:
    return os.getenv("NEWS_LABELER_BASE_URL") or provider_config.get("base_url", "")


def get_labeling_config(config: dict = None) -> dict:
    config = config or load_providers_config()
    return config.get("labeling", {
        "temperature": 0.0,
        "max_tokens": 10,
        "batch_size": 10,
    })
