"""API routes for the news classifier webhook."""

from tradingagents.news_classifier.webhook.server import create_app

app = create_app()
