from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableLambda

from tradingagents.agents.analysts.narrative_analyst import (
    create_market_narrative_analyst,
    create_narrative_analyst,
    create_narrative_picker,
    create_news_narrative_analyst,
    create_social_narrative_analyst,
    route_narrative_entry,
)


class NarrativeAgentTests(unittest.TestCase):
    def test_route_narrative_entry(self):
        # Case A: User narrative present -> skip_search
        state_with_user = {"user_narrative": "Inflation is falling"}
        self.assertEqual(route_narrative_entry(state_with_user), "skip_search")

        # Case B: User narrative empty -> run_search
        state_empty = {"user_narrative": ""}
        self.assertEqual(route_narrative_entry(state_empty), "run_search")

        # Case C: User narrative missing -> run_search
        state_missing = {}
        self.assertEqual(route_narrative_entry(state_missing), "run_search")

    @patch("tradingagents.agents.analysts.narrative_analyst.get_news")
    @patch("tradingagents.agents.analysts.narrative_analyst.get_global_news")
    @patch("tradingagents.agents.analysts.narrative_analyst.get_web_search")
    def test_news_narrative_analyst(self, mock_web_search, mock_global_news, mock_news):
        mock_news.invoke.return_value = "Mocked News Headlines"
        mock_global_news.invoke.return_value = "Mocked Global News"
        mock_web_search.invoke.return_value = "Mocked Web Search"

        llm = RunnableLambda(lambda x: MagicMock(content="Prevalent News Narrative: AI chip boom"))

        node = create_news_narrative_analyst(llm)
        state = {
            "trade_date": "2026-07-07",
            "company_of_interest": "AAPL",
            "messages": [],
        }

        result = node(state)
        self.assertEqual(result["news_narrative"], "Prevalent News Narrative: AI chip boom")
        mock_news.invoke.assert_called_once()
        mock_global_news.invoke.assert_called_once()
        mock_web_search.invoke.assert_called_once()

    @patch("tradingagents.agents.analysts.narrative_analyst.fetch_stocktwits_messages")
    @patch("tradingagents.agents.analysts.narrative_analyst.fetch_reddit_posts")
    @patch("tradingagents.agents.analysts.narrative_analyst.fetch_bluesky_posts")
    @patch("tradingagents.agents.analysts.narrative_analyst.fetch_mastodon_posts")
    @patch("tradingagents.agents.analysts.narrative_analyst.get_fear_greed_index")
    def test_social_narrative_analyst(
        self, mock_fg, mock_mastodon, mock_bluesky, mock_reddit, mock_st
    ):
        mock_st.return_value = "StockTwits messages"
        mock_reddit.return_value = "Reddit posts"
        mock_bluesky.return_value = "Bluesky posts"
        mock_mastodon.return_value = "Mastodon posts"
        mock_fg.return_value = "Fear & Greed Index"

        llm = RunnableLambda(
            lambda x: MagicMock(content="Prevalent Social Narrative: Buy the dip retail hype")
        )

        node = create_social_narrative_analyst(llm)
        state = {
            "trade_date": "2026-07-07",
            "company_of_interest": "AAPL",
            "messages": [],
        }

        result = node(state)
        self.assertEqual(
            result["social_narrative"], "Prevalent Social Narrative: Buy the dip retail hype"
        )
        mock_st.assert_called_once()
        mock_reddit.assert_called_once()

    @patch("tradingagents.agents.analysts.narrative_analyst.get_stock_data")
    @patch("tradingagents.agents.analysts.narrative_analyst.get_indicators")
    def test_market_narrative_analyst(self, mock_indicators, mock_stock_data):
        mock_stock_data.invoke.return_value = "Mocked Stock Price Data"
        mock_indicators.invoke.return_value = "Mocked indicator value"

        llm = RunnableLambda(
            lambda x: MagicMock(content="Prevalent Market Narrative: Parabolic rally")
        )

        node = create_market_narrative_analyst(llm)
        state = {
            "trade_date": "2026-07-07",
            "company_of_interest": "AAPL",
            "messages": [],
        }

        result = node(state)
        self.assertEqual(result["market_narrative"], "Prevalent Market Narrative: Parabolic rally")
        mock_stock_data.invoke.assert_called_once()
        self.assertEqual(mock_indicators.invoke.call_count, 3)

    def test_narrative_picker_user_narrative(self):
        llm = RunnableLambda(
            lambda x: MagicMock(content="Narrative Report: Refined inflation narrative")
        )

        node = create_narrative_picker(llm)
        state = {
            "trade_date": "2026-07-07",
            "company_of_interest": "AAPL",
            "messages": [],
            "user_narrative": "Inflation is falling",
        }

        result = node(state)
        self.assertEqual(
            result["narrative_report"], "Narrative Report: Refined inflation narrative"
        )

    def test_narrative_picker_agent_narrative(self):
        llm = RunnableLambda(
            lambda x: MagicMock(content="Narrative Report: Combined structural AI narrative")
        )

        node = create_narrative_picker(llm)
        state = {
            "trade_date": "2026-07-07",
            "company_of_interest": "AAPL",
            "messages": [],
            "user_narrative": "",
            "news_narrative": "AI growth",
            "social_narrative": "Bullish speculation",
            "market_narrative": "Oversold bounce",
        }

        result = node(state)
        self.assertEqual(
            result["narrative_report"], "Narrative Report: Combined structural AI narrative"
        )

    def test_dummy_narrative_analyst_compat(self):
        llm = MagicMock()
        node = create_narrative_analyst(llm)
        state = {
            "trade_date": "2026-07-07",
            "company_of_interest": "AAPL",
            "messages": [],
        }
        # Dummy node returns empty dict
        result = node(state)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
