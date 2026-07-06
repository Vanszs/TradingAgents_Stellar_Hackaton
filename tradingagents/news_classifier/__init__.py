"""Crypto News Classifier - Real-time news importance classification."""


def get_classifier():
    from tradingagents.news_classifier.inference.classifier import NewsClassifier
    return NewsClassifier


def get_sanitizer():
    from tradingagents.news_classifier.inference.sanitizer import NewsSanitizer
    return NewsSanitizer


__all__ = ["get_classifier", "get_sanitizer"]
