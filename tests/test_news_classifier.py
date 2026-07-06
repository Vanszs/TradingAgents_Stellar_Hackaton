"""Tests for the crypto news classifier."""

import pytest
import sys
from unittest.mock import patch, MagicMock

from tradingagents.news_classifier.data.preprocessor import (
    clean_html,
    normalize_whitespace,
    normalize_crypto_terms,
    remove_urls,
    preprocess_article,
)


class TestPreprocessor:
    def test_clean_html(self):
        result = clean_html("<p>Hello &amp; world</p>").strip()
        assert "Hello" in result
        assert "&" in result
        assert "world" in result
        result2 = clean_html("<b>Bold</b> text").strip()
        assert "Bold" in result2
        assert "text" in result2

    def test_normalize_whitespace(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"
        assert normalize_whitespace("multi\n\nline") == "multi line"

    def test_normalize_crypto_terms(self):
        text = "BTC and ETH are popular DeFi tokens"
        result = normalize_crypto_terms(text)
        assert "Bitcoin" in result
        assert "Ethereum" in result
        assert "Decentralized Finance" in result

    def test_remove_urls(self):
        text = "Check https://example.com for details"
        result = remove_urls(text)
        assert "https" not in result
        assert "details" in result

    def test_preprocess_article(self):
        result = preprocess_article(
            title="BTC surges",
            content="<p>Bitcoin price hit $100k</p>",
        )
        assert "TITLE:" in result
        assert "Bitcoin" in result
        assert "<p>" not in result


from tradingagents.news_classifier.config import LABEL_MAP, ID_TO_LABEL


class TestConfig:
    def test_label_map(self):
        assert LABEL_MAP["BIASA"] == 0
        assert LABEL_MAP["LUMAYAN"] == 1
        assert LABEL_MAP["PENTING"] == 2

    def test_id_to_label(self):
        assert ID_TO_LABEL[0] == "BIASA"
        assert ID_TO_LABEL[1] == "LUMAYAN"
        assert ID_TO_LABEL[2] == "PENTING"


from tradingagents.news_classifier.training.evaluator import compute_metrics, confusion_matrix


class TestEvaluator:
    def test_compute_metrics_perfect(self):
        preds = [0, 1, 2, 0, 1, 2]
        labels = [0, 1, 2, 0, 1, 2]
        metrics = compute_metrics(preds, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_compute_metrics_imperfect(self):
        preds = [0, 0, 0]
        labels = [0, 1, 2]
        metrics = compute_metrics(preds, labels)
        assert metrics["accuracy"] < 1.0
        assert 0 <= metrics["f1_macro"] <= 1

    def test_confusion_matrix(self):
        preds = [0, 1, 2, 0]
        labels = [0, 1, 2, 1]
        matrix = confusion_matrix(preds, labels)
        assert matrix[0][0] == 1
        assert matrix[1][0] == 1
        assert matrix[1][1] == 1
        assert matrix[2][2] == 1


from tradingagents.news_classifier.training.augmentor import (
    synonym_replacement,
    random_deletion,
    random_swap,
    augment_text,
)


class TestAugmentor:
    def test_synonym_replacement(self):
        text = "The market crash caused panic"
        result = synonym_replacement(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_random_deletion(self):
        text = "This is a test sentence with multiple words"
        result = random_deletion(text, p=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_random_swap(self):
        text = "one two three four five"
        result = random_swap(text)
        assert isinstance(result, str)
        assert sorted(result.split()) == sorted(text.split())

    def test_augment_text(self):
        text = "Bitcoin crashes after regulatory news"
        results = augment_text(text, num_augmentations=3)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


from tradingagents.news_classifier.webhook.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ActiveCoinsConfig,
    ActiveCoin,
)


class TestSchemas:
    def test_classify_request(self):
        req = ClassifyRequest(title="Test", content="Content", source="RSS")
        assert req.title == "Test"

    def test_classify_response(self):
        resp = ClassifyResponse(
            label="PENTING",
            confidence=0.95,
            probabilities={"BIASA": 0.01, "LUMAYAN": 0.04, "PENTING": 0.95},
            title="Test",
            source="RSS",
        )
        assert resp.label == "PENTING"

    def test_active_coins_config(self):
        config = ActiveCoinsConfig(coins=[
            ActiveCoin(ticker="BTC", name="Bitcoin", narratives=["ETF"]),
        ])
        assert len(config.coins) == 1
        assert config.coins[0].ticker == "BTC"
