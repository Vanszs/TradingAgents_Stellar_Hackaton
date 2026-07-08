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

LABEL_MAP = {"NORMAL": 0, "MODERATE": 1, "CRITICAL": 2}
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

GOOGLE_NEWS_QUERIES = [
    # ===== CRITICAL (50 queries) - Portfolio Impact Langsung =====
    # Existing CRITICAL (30)
    "crypto exchange hack stolen",
    "crypto exchange collapse bankruptcy",
    "Bitcoin ETF SEC rejection",
    "crypto regulation ban country",
    "SEC crypto enforcement lawsuit",
    "stablecoin depeg USDT USDC",
    "DeFi exploit hack million",
    "crypto exchange delisting token",
    "Bitcoin whale dump sell million",
    "crypto market crash today",
    "Tether USDT bankruptcy risk",
    "crypto exchange frozen withdrawal",
    "Binance regulatory action SEC",
    "crypto Ponzi scheme fraud",
    "Ethereum critical vulnerability",
    "Bitcoin mining ban regulation",
    "crypto tax regulation crackdown",
    "DeFi protocol insolvency",
    "crypto exchange liquidity crisis",
    "stablecoin reserve audit failure",
    "major crypto project rug pull",
    "crypto lending platform collapse",
    "Bitcoin futures liquidation cascade",
    "crypto custody failure loss",
    "nation state crypto ban announcement",
    "crypto exchange hack",
    "Bitcoin ETF SEC approval",
    "crypto exchange collapse bankruptcy",
    "DeFi exploit hack million",
    "crypto regulation ban",
    # Migrated from MODERATE (5)
    "Solana network outage downtime",
    "Ethereum upgrade merge hard fork",
    "crypto ETF approval speculation",
    "Bitcoin futures liquidation cascade",
    "Bitcoin hash rate difficulty change",
    # Sector → CRITICAL (15)
    "AI regulation impact crypto tokens",
    "gaming regulation crypto ban",
    "Monero XMR privacy coin regulation",
    "privacy coin ban regulation news",
    "memecoin pump dump whale alert",
    "DeFi protocol hack exploit news",
    "crypto gaming ban regulation news",
    "AI crypto security vulnerability news",
    "Dogecoin DOGE Elon Musk news",
    "MakerDAO MKR stablecoin DAI news",
    "crypto staking platform insolvency",
    "crypto lending rate crash news",
    "DeFi oracle manipulation attack",
    "crypto bridge exploit hack news",
    "stablecoin algorithmic failure news",

    # ===== MODERATE (41 queries) - Portfolio Influence =====
    "Bitcoin price analysis forecast",
    "crypto institutional adoption purchase",
    "DeFi TVL total value locked change",
    "crypto exchange new listing token",
    "Bitcoin halving impact price",
    "crypto venture capital funding",
    "altcoin season rotation market",
    "crypto market sentiment fear greed",
    "Ethereum gas fee spike high",
    "crypto derivatives options expiry",
    "DeFi yield farming opportunity",
    "crypto market correlation stocks",
    "stablecoin market share competition",
    "crypto regulatory framework clarity",
    "NFT market volume trading",
    "crypto mining profitability change",
    "layer 2 scaling solution adoption",
    "crypto cross-border payment adoption",
    "blockchain enterprise implementation",
    "crypto derivatives volume open interest",
    "crypto index fund rebalancing",
    # Sector → MODERATE (19)
    "AI crypto token artificial intelligence news",
    "Fetch.ai partnership announcement",
    "Render Network GPU computing news",
    "Bittensor TAO AI mining news",
    "Ocean Protocol data marketplace news",
    "AI agent crypto autonomous trading",
    "crypto gaming token play to earn news",
    "Sandbox SAND metaverse news",
    "blockchain game launch new token",
    "metaverse partnership major brand",
    "Chainlink LINK oracle partnership",
    "DeFi total value locked TVL surge",
    "DeFi yield farming opportunity news",
    "Polygon MATIC scaling partnership news",
    "Arbitrum ARB ecosystem growth news",
    "Filecoin FIL storage adoption news",
    "Chainlink LINK price feed integration",
    "enterprise blockchain adoption news",
    "crypto governance proposal UNI AAVE",

    # ===== NORMAL (17 queries) - Info Only =====
    "crypto market news today",
    "Bitcoin price movement daily",
    "crypto community sentiment twitter",
    "crypto developer activity github",
    "crypto podcast interview analysis",
    "crypto education learning guide",
    "crypto wallet security tips",
    "crypto regulatory update minor",
    "crypto market data chart analysis",
    "crypto project roadmap update",
    "crypto job market hiring",
    "crypto conference event announcement",
    "crypto research paper published",
    "crypto social media trend viral",
    "crypto trading strategy tips",
    # Sector → NORMAL (2)
    "SingularityNET AGIX development update",
    "Decentraland MANA virtual world update",
]

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

CRYPTO_KEYWORDS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "AVAX",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "ETC", "FIL", "APT", "ARB",
    "OP", "NEAR", "ICP", "FTM", "ALGO", "XLM", "VET", "HBAR", "MANA",
    "SAND", "AXS", "GALA", "BITCOIN", "ETHEREUM", "SOLANA", "BINANCE",
    "RIPPLE", "CARDANO", "DOGECOIN", "POLKADOT", "AVALANCHE", "POLYGON",
    "CHAINLINK", "UNISWAP", "COSMOS", "LITECOIN",
}

SANITIZER_PROMPT = """You are a crypto news impact assessor. A machine learning model classified this news as "CRITICAL".

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

- CRITICAL: Highly impactful news that could significantly move crypto markets. Examples: major exchange hack/collapse, ETF approval/rejection, major regulatory action, huge whale movements, protocol critical vulnerability, major partnership listing.
- MODERATE: Moderately impactful news. Examples: minor exchange issues, project updates, moderate regulatory news, market analysis with notable insights, minor hack/exploit.
- NORMAL: Regular/routine news. Examples: price updates, routine project announcements, opinion pieces, minor market commentary.

Respond with ONLY the classification label: CRITICAL, MODERATE, or NORMAL.

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
