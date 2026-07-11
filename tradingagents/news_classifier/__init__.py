"""Crypto News Classifier - Real-time news importance classification."""


def get_classifier():
    from tradingagents.news_classifier.inference.classifier import NewsClassifier
    return NewsClassifier


__all__ = ["get_classifier"]
