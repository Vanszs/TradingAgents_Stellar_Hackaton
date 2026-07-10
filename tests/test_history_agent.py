from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from tradingagents.agents.history_agent import create_history_agent
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.utils.report_history import discover_latest_ticker_report


def _prompt_capture_llm(captured: dict):
    llm = MagicMock()

    def invoke(prompt, config=None, **kwargs):
        captured["prompt"] = prompt
        return AIMessage(content="History conclusion.")

    llm.invoke.side_effect = invoke
    return llm


def _structured_prompt_capture_llm(captured: dict):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt)
        or PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold for now.",
            investment_thesis="The debate remains balanced.",
        )
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def test_discover_latest_ticker_report_selects_newest(tmp_path):
    project_dir = tmp_path
    older = project_dir / "backtest_results" / "BREN.JK_20260523_1" / "report.md"
    newer = project_dir / "backtest_results" / "BREN.JK_20260523_2" / "report.md"
    older.parent.mkdir(parents=True, exist_ok=True)
    newer.parent.mkdir(parents=True, exist_ok=True)
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    older_ts = 1_700_000_000
    newer_ts = 1_800_000_000
    older.touch()
    newer.touch()
    Path(older).touch()
    Path(newer).touch()
    import os

    os.utime(older, (older_ts, older_ts))
    os.utime(newer, (newer_ts, newer_ts))

    result = discover_latest_ticker_report("BREN.JK", str(project_dir))

    assert result == newer


def test_history_agent_uses_report_and_returns_summary(tmp_path):
    project_dir = tmp_path
    report_path = project_dir / "backtest_results" / "BREN.JK" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("Final Equity: 89757748.80\nTotal Return: -10.24%\n", encoding="utf-8")

    captured = {}
    llm = _prompt_capture_llm(captured)
    node = create_history_agent(llm, str(project_dir))

    result = node({"company_of_interest": "BREN.JK", "asset_type": "stock"})

    assert result["historical_summary"] == "History conclusion."
    assert "Final Equity: 89757748.80" in captured["prompt"]
    assert "BREN.JK" in captured["prompt"]


def test_history_agent_falls_back_when_report_missing(tmp_path):
    captured = {}
    llm = _prompt_capture_llm(captured)
    node = create_history_agent(llm, str(tmp_path))

    result = node({"company_of_interest": "BREN.JK", "asset_type": "stock"})

    assert "No historical backtest report was found" in result["historical_summary"]
    assert "prompt" not in captured


def test_research_prompts_include_history_summary():
    shared_state = {
        "company_of_interest": "BREN.JK",
        "asset_type": "stock",
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "narrative_report": "Narrative report.",
        "historical_summary": "Ticker drifted from strong gains to a deep drawdown.",
        "investment_debate_state": {
            "history": "Debate history.",
            "bull_history": "",
            "bear_history": "",
            "current_response": "Bear Analyst: ...",
            "judge_decision": "",
            "count": 1,
        },
    }

    bull_capture = {}
    bull_node = create_bull_researcher(_prompt_capture_llm(bull_capture))
    bull_node(shared_state)
    assert "Ticker drifted from strong gains" in bull_capture["prompt"]

    bear_capture = {}
    bear_node = create_bear_researcher(_prompt_capture_llm(bear_capture))
    bear_node(shared_state)
    assert "Ticker drifted from strong gains" in bear_capture["prompt"]

    manager_capture = {}
    manager_node = create_research_manager(_structured_prompt_capture_llm(manager_capture))
    manager_node(shared_state)
    assert "Ticker drifted from strong gains" in manager_capture["prompt"]
