"""Fast inference pipeline for crypto news classification."""

import logging
import re
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import torch
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
)

from tradingagents.news_classifier.config import (
    MODEL_NAME,
    MAX_LENGTH,
    ID_TO_LABEL,
    PRETRAINED_DIR,
)
from tradingagents.news_classifier.data.preprocessor import preprocess_with_features

logger = logging.getLogger(__name__)

PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)


def _clean_source_name(source: str) -> str:
    if not source:
        return ""
    if not source.startswith("http"):
        if ":" in source:
            return source.split(":", 1)[1].strip()
        return source.strip()
    try:
        parsed = urlparse(source)
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split(".")[0]
        return domain.capitalize()
    except Exception:
        return source


class NewsClassifier:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = MAX_LENGTH,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model_name = model_name

        if model_path is None:
            model_path = PRETRAINED_DIR / "best_model.pt"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if Path(model_path).exists():
            logger.info("Loading trained model from %s", model_path)
            state_dict = torch.load(str(model_path), map_location=self.device)

            # Auto-detect model architecture
            has_roberta = any(k.startswith("roberta.") for k in state_dict.keys())
            has_albert = any(k.startswith("albert.") for k in state_dict.keys())

            if has_roberta:
                # Create model with correct config
                config = RobertaConfig.from_pretrained("roberta-base")
                config.num_labels = 3
                self.model = RobertaForSequenceClassification(config)

                # Load state dict directly (keys already match)
                self.model.load_state_dict(state_dict)
                logger.info("Loaded RoBERTa model with trained classifier")
            elif has_albert:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "albert-base-v2", num_labels=3
                )
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded ALBERT model")
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=3
                )
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded model with default architecture")

            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully from %s", model_path)
        else:
            logger.warning("No trained model at %s, using untrained %s", model_path, model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=3
            )
            self.model.to(self.device)
            self.model.eval()

    def classify(
        self, title: str, content: str = "", source: str = "",
        url: str = "", pub_date: str = "", description: str = "",
    ) -> dict:
        article = {
            "title": title,
            "description": content,
            "source": source,
        }
        text = preprocess_with_features(article)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()

        label = ID_TO_LABEL[pred_idx]

        return {
            "label": label,
            "label_id": pred_idx,
            "confidence": round(confidence, 4),
            "probabilities": {
                ID_TO_LABEL[i]: round(probs[0][i].item(), 4)
                for i in range(len(ID_TO_LABEL))
            },
            "title": title,
            "description": description or content,
            "url": url,
            "pub_date": pub_date,
            "source": _clean_source_name(source),
        }

    def classify_batch(self, articles: list[dict]) -> list[dict]:
        results = []
        for article in articles:
            result = self.classify(
                title=article.get("title", ""),
                content=article.get("description", article.get("content", "")),
                source=article.get("source", ""),
                url=article.get("link", ""),
                pub_date=article.get("pub_date", ""),
                description=article.get("description", ""),
            )
            results.append(result)
        return results

    def is_critical(self, title: str, content: str = "", threshold: float = 0.6) -> bool:
        result = self.classify(title, content)
        return result["label"] == "CRITICAL" and result["confidence"] >= threshold

