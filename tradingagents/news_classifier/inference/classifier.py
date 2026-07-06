"""Fast inference pipeline for crypto news classification."""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoTokenizer

from tradingagents.news_classifier.config import (
    MODEL_NAME,
    MAX_LENGTH,
    ID_TO_LABEL,
    PRETRAINED_DIR,
)
from tradingagents.news_classifier.models.bert_classifier import CryptoNewsBERT, load_model, load_tokenizer
from tradingagents.news_classifier.data.preprocessor import preprocess_article

logger = logging.getLogger(__name__)


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

        if model_path is None:
            model_path = PRETRAINED_DIR / "best_model.pt"

        self.tokenizer = load_tokenizer(model_name)

        if Path(model_path).exists():
            self.model = load_model(str(model_path), model_name=model_name, device=self.device)
            logger.info("Loaded trained model from %s", model_path)
        else:
            logger.warning("No trained model found at %s, using untrained model", model_path)
            self.model = CryptoNewsBERT(model_name=model_name)
            self.model.to(self.device)
            self.model.eval()

    def classify(self, title: str, content: str = "", source: str = "") -> dict:
        text = preprocess_article(title=title, content=content)

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
            probs = torch.softmax(outputs["logits"], dim=-1)
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
            "source": source,
        }

    def classify_batch(self, articles: list[dict]) -> list[dict]:
        results = []
        for article in articles:
            result = self.classify(
                title=article.get("title", ""),
                content=article.get("description", article.get("content", "")),
                source=article.get("source", ""),
            )
            results.append(result)
        return results

    def is_important(self, title: str, content: str = "", threshold: float = 0.7) -> bool:
        result = self.classify(title, content)
        return result["label"] == "PENTING" and result["confidence"] >= threshold
