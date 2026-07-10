"""BERT-based classifier for crypto news importance."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from tradingagents.news_classifier.config import MODEL_NAME, NUM_LABELS, MAX_LENGTH

logger = logging.getLogger(__name__)


class CryptoNewsBERT(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.3,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("Frozen base model parameters")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return torch.argmax(outputs["logits"], dim=-1)

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return torch.softmax(outputs["logits"], dim=-1)


def load_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def load_model(
    checkpoint_path: Optional[str] = None,
    model_name: str = MODEL_NAME,
    num_labels: int = NUM_LABELS,
    device: Optional[str] = None,
) -> CryptoNewsBERT:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CryptoNewsBERT(model_name=model_name, num_labels=num_labels)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Loaded model from %s", checkpoint_path)

    model.to(device)
    model.eval()
    return model
