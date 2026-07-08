from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_indicators,
    get_language_instruction,
    get_news,
    get_stock_data,
)
from tradingagents.agents.utils.web_search_tools import get_web_search
from tradingagents.dataflows.bluesky import fetch_bluesky_posts
from tradingagents.dataflows.fear_greed import get_fear_greed_index
from tradingagents.dataflows.mastodon import fetch_mastodon_posts
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages


def route_narrative_entry(state):
    """Conditional routing edge for narrative analysis.

    If the user has provided a custom narrative, skip narrative searches and
    go straight to the picker to detail the user's narrative.
    Otherwise, run news, social, and market narrative searches in parallel.
    """
    user_narrative = state.get("user_narrative")
    if user_narrative and user_narrative.strip():
        return "skip_search"
    return "run_search"


def create_news_narrative_analyst(llm):
    """News-based narrative searcher agent node."""

    def news_narrative_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(ticker, asset_type=asset_type)

        # Calculate dates for Yahoo Finance news fetch
        end_date = current_date
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=7)).strftime(
            "%Y-%m-%d"
        )

        # Fetch news sources
        news_raw = get_news.invoke(
            {"ticker": ticker, "start_date": start_date, "end_date": end_date}
        )
        global_news_raw = get_global_news.invoke({"curr_date": current_date})
        web_search_raw = get_web_search.invoke(
            {"query": f"{ticker} latest narrative market impact {current_date}"}
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a News-based Narrative Analyst. Your task is to identify and recommend the single most prevalent market or industry narrative currently affecting `{ticker}` from the news media.\n"
                    "Use the following gathered news and trends to determine the narrative:\n\n"
                    "### Yahoo Finance News:\n{news_raw}\n\n"
                    "### Global Macro News:\n{global_news_raw}\n\n"
                    "### Web Search Context:\n{web_search_raw}\n\n"
                    "Format your response as a concise summary of this narrative, highlighting key facts, dates, and sources. Keep it brief (under 3 paragraphs).\n"
                    "Current analysis date: {current_date}. {instrument_context}"
                    + get_language_instruction(),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = (
            prompt.partial(
                ticker=ticker,
                news_raw=news_raw,
                global_news_raw=global_news_raw,
                web_search_raw=web_search_raw,
                current_date=current_date,
                instrument_context=instrument_context,
            )
            | llm
        )

        result = chain.invoke({"messages": state["messages"]})
        return {
            "messages": [result],
            "news_narrative": result.content,
        }

    return news_narrative_analyst_node


def create_social_narrative_analyst(llm):
    """Social media-based narrative searcher agent node."""

    def social_narrative_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(ticker, asset_type=asset_type)

        # Fetch StockTwits
        if asset_type == "crypto":
            st_ticker = ticker.split("-")[0] + ".X"
        else:
            st_ticker = ticker
        stocktwits_raw = fetch_stocktwits_messages(st_ticker, limit=30)

        # Fetch Reddit
        if asset_type == "crypto":
            crypto_subs = ("CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin")
            reddit_raw = fetch_reddit_posts(ticker.split("-")[0], subreddits=crypto_subs)
        else:
            reddit_raw = fetch_reddit_posts(ticker)

        # Fetch Bluesky & Mastodon
        base = ticker.split("-")[0] if asset_type == "crypto" else ticker
        bluesky_raw = fetch_bluesky_posts(f"${base}")
        mastodon_raw = fetch_mastodon_posts(base)
        fear_greed_raw = get_fear_greed_index()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Social Media-based Narrative Analyst. Your task is to identify and recommend the single most prevalent market/industry narrative, trend, or rumor circulating on social media regarding `{ticker}`.\n"
                    "Use the following social media discussions and sentiment data:\n\n"
                    "### StockTwits Messages:\n{stocktwits_raw}\n\n"
                    "### Reddit Posts:\n{reddit_raw}\n\n"
                    "### Bluesky Posts:\n{bluesky_raw}\n\n"
                    "### Mastodon Posts:\n{mastodon_raw}\n\n"
                    "### Fear & Greed Index:\n{fear_greed_raw}\n\n"
                    "Format your response as a concise summary of this social narrative, highlighting retail sentiment and popular community discussion points. Keep it brief (under 3 paragraphs).\n"
                    "Current analysis date: {current_date}. {instrument_context}"
                    + get_language_instruction(),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = (
            prompt.partial(
                ticker=ticker,
                stocktwits_raw=stocktwits_raw,
                reddit_raw=reddit_raw,
                bluesky_raw=bluesky_raw,
                mastodon_raw=mastodon_raw,
                fear_greed_raw=fear_greed_raw,
                current_date=current_date,
                instrument_context=instrument_context,
            )
            | llm
        )

        result = chain.invoke({"messages": state["messages"]})
        return {
            "messages": [result],
            "social_narrative": result.content,
        }

    return social_narrative_analyst_node


def create_market_narrative_analyst(llm):
    """Market/technical-based narrative searcher agent node."""

    def market_narrative_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(ticker, asset_type=asset_type)

        # Calculate dates for stock data fetch
        start_date = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=60)).strftime(
            "%Y-%m-%d"
        )
        stock_data_raw = get_stock_data.invoke(
            {"symbol": ticker, "start_date": start_date, "end_date": current_date}
        )

        # Fetch Indicators
        macd_raw = get_indicators.invoke(
            {"symbol": ticker, "indicator": "macd", "curr_date": current_date, "look_back_days": 60}
        )
        rsi_raw = get_indicators.invoke(
            {"symbol": ticker, "indicator": "rsi", "curr_date": current_date, "look_back_days": 30}
        )
        boll_raw = get_indicators.invoke(
            {"symbol": ticker, "indicator": "boll", "curr_date": current_date, "look_back_days": 30}
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Market-based Narrative Analyst. Your task is to identify and recommend the dominant market narrative indicated by the price structure, technical setups, or volume trends for `{ticker}` (e.g. 'oversold capitulation bounce', 'parabolic momentum breakout', 'liquidity accumulation phase').\n"
                    "Use the following market indicators and data:\n\n"
                    "### Historical Stock Data (last 60 days):\n{stock_data_raw}\n\n"
                    "### Technical MACD Indicator:\n{macd_raw}\n\n"
                    "### Technical RSI Indicator:\n{rsi_raw}\n\n"
                    "### Bollinger Bands Indicator:\n{boll_raw}\n\n"
                    "Format your response as a concise summary of this market technical narrative and the key setups supporting it. Keep it brief (under 3 paragraphs).\n"
                    "Current analysis date: {current_date}. {instrument_context}"
                    + get_language_instruction(),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = (
            prompt.partial(
                ticker=ticker,
                stock_data_raw=stock_data_raw,
                macd_raw=macd_raw,
                rsi_raw=rsi_raw,
                boll_raw=boll_raw,
                current_date=current_date,
                instrument_context=instrument_context,
            )
            | llm
        )

        result = chain.invoke({"messages": state["messages"]})
        return {
            "messages": [result],
            "market_narrative": result.content,
        }

    return market_narrative_analyst_node


def create_narrative_picker(llm):
    """Narrative Picker agent node."""

    def narrative_picker_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(ticker, asset_type=asset_type)
        user_narrative = state.get("user_narrative")

        news_narrative = state.get("news_narrative", "")
        social_narrative = state.get("social_narrative", "")
        market_narrative = state.get("market_narrative", "")

        if user_narrative and user_narrative.strip():
            system_message = (
                f"You are a narrative analyst picker. The user has specified the following narrative for `{ticker}`:\n"
                f"'{user_narrative}'\n\n"
                f"Your task is to refine and detail this narrative specifically for `{ticker}`. "
                "Explain what the narrative represents, its macroeconomic or industry relevance, and its general potential impact on the company/asset. "
                "Output a comprehensive report (the Narrative Report) focusing on this narrative, its implications, and relevance."
                + get_language_instruction()
            )
        else:
            system_message = (
                f"You are a narrative analyst picker. Your task is to evaluate three candidate narratives identified by specialized searcher agents for `{ticker}`:\n\n"
                f"### 1. News Media Narrative:\n{news_narrative}\n\n"
                f"### 2. Social Media Narrative:\n{social_narrative}\n\n"
                f"### 3. Market/Price Action Narrative:\n{market_narrative}\n\n"
                "From these three candidate narratives, select at least one (you may select more than one if they are highly relevant and compatible) that affects the ticker most.\n"
                "Write a comprehensive report (the Narrative Report) detailing this/these chosen narrative(s), explaining your reasoning for the selection(s), and detailing the positive/negative implications for `{ticker}`.\n"
                + get_language_instruction()
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants. "
                    "{system_message}\n\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = (
            prompt.partial(
                system_message=system_message,
                current_date=current_date,
                instrument_context=instrument_context,
            )
            | llm
        )

        result = chain.invoke({"messages": state["messages"]})
        return {
            "messages": [result],
            "narrative_report": result.content,
        }

    return narrative_picker_node


def create_narrative_analyst(llm):
    """The original Narrative Analyst node name.

    Serves as the entry/delegator node in the StateGraph for backwards compatibility.
    Does not run any LLM calls itself, simply returning state unchanged. Routing logic
    is handled by `route_narrative_entry`.
    """

    def narrative_entry_node(state):
        return {}

    return narrative_entry_node
