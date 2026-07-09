"""Fast inference pipeline for crypto news classification."""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tradingagents.news_classifier.config import (
    MODEL_NAME,
    MAX_LENGTH,
    ID_TO_LABEL,
    PRETRAINED_DIR,
)
from tradingagents.news_classifier.data.preprocessor import preprocess_with_features

logger = logging.getLogger(__name__)

PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)


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
            state_dict = torch.load(str(model_path), map_location=self.device)

            # Check which keys are in state dict
            has_albert = any(k.startswith("albert.") for k in state_dict.keys())
            has_roberta = any(k.startswith("roberta.") for k in state_dict.keys())
            has_distilbert = any(k.startswith("distilbert.") for k in state_dict.keys())
            has_bert = any(k.startswith("bert.") for k in state_dict.keys())

            # Try to load with the correct model - skip if download fails
            model_loaded = False
            attempts = []

            if has_albert:
                attempts.append(("albert-base-v2", "ALBERT"))
            if has_roberta:
                attempts.append(("roberta-base", "RoBERTa"))
            if has_distilbert:
                attempts.append(("distilbert-base-uncased", "DistilBERT"))
            if has_bert or not attempts:
                attempts.append((model_name, "BERT"))

            for model_id, model_type in attempts:
                try:
                    logger.info("Attempting to load %s from %s...", model_type, model_id)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_id, num_labels=3
                    )
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("Successfully loaded %s model", model_type)
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning("Failed to load %s (%s), trying next...", model_type, str(e)[:100])
                    continue

            if not model_loaded:
                logger.warning("No model loaded, using untrained %s", model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=3
                )
                self.model.to(self.device)
                self.model.eval()
        else:
            logger.warning("No trained model found at %s, using untrained model", model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=3
            )
            self.model.to(self.device)
            self.model.eval()

    def classify(self, title: str, content: str = "", source: str = "") -> dict:
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

    def is_critical(self, title: str, content: str = "", threshold: float = 0.6) -> bool:
        result = self.classify(title, content)
        return result["label"] == "CRITICAL" and result["confidence"] >= threshold
