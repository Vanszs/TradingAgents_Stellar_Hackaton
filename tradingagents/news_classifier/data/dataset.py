"""PyTorch Dataset for crypto news classification."""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from tradingagents.news_classifier.config import LABEL_MAP, MAX_LENGTH
from tradingagents.news_classifier.data.preprocessor import preprocess_article

logger = logging.getLogger(__name__)


class CryptoNewsDataset(Dataset):
    def __init__(
        self,
        data_path: Optional[Path] = None,
        articles: Optional[list[dict]] = None,
        tokenizer=None,
        max_length: int = MAX_LENGTH,
        normalize_crypto: bool = True,
    ):
        self.max_length = max_length
        self.normalize_crypto = normalize_crypto
        self.tokenizer = tokenizer

        if articles is not None:
            self.articles = articles
        elif data_path is not None:
            self.articles = self._load_from_file(data_path)
        else:
            raise ValueError("Provide either data_path or articles")

        self.articles = [a for a in self.articles if "label" in a and a["label"] in LABEL_MAP]
        logger.info("Dataset loaded: %d labeled articles", len(self))

    def _load_from_file(self, path: Path) -> list[dict]:
        articles = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    articles.append(json.loads(line))
        return articles

    def __len__(self) -> int:
        return len(self.articles)

    def __getitem__(self, idx: int) -> dict:
        article = self.articles[idx]
        text = preprocess_article(
            title=article.get("title", ""),
            content=article.get("description", ""),
            normalize_crypto=self.normalize_crypto,
            max_words=400,
        )
        label = LABEL_MAP[article["label"]]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
                "text": text,
                "title": article.get("title", ""),
            }

        return {
            "text": text,
            "labels": torch.tensor(label, dtype=torch.long),
            "title": article.get("title", ""),
        }

    def get_class_weights(self) -> torch.Tensor:
        counts = [0] * len(LABEL_MAP)
        for article in self.articles:
            label = LABEL_MAP[article["label"]]
            counts[label] += 1

        total = sum(counts)
        weights = [total / (len(LABEL_MAP) * max(c, 1)) for c in counts]
        return torch.tensor(weights, dtype=torch.float32)


def load_dataset(
    data_path: Path,
    tokenizer=None,
    max_length: int = MAX_LENGTH,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[CryptoNewsDataset, CryptoNewsDataset, CryptoNewsDataset]:
    full_dataset = CryptoNewsDataset(data_path=data_path, tokenizer=tokenizer, max_length=max_length)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()

    train_end = int(len(indices) * train_ratio)
    val_end = int(len(indices) * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_articles = [full_dataset.articles[i] for i in train_indices]
    val_articles = [full_dataset.articles[i] for i in val_indices]
    test_articles = [full_dataset.articles[i] for i in test_indices]

    train_ds = CryptoNewsDataset(articles=train_articles, tokenizer=tokenizer, max_length=max_length)
    val_ds = CryptoNewsDataset(articles=val_articles, tokenizer=tokenizer, max_length=max_length)
    test_ds = CryptoNewsDataset(articles=test_articles, tokenizer=tokenizer, max_length=max_length)

    logger.info("Split: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds
