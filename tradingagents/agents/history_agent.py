from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.report_history import discover_latest_ticker_report


def create_history_agent(llm, project_dir: str):
    def history_node(state) -> dict:
        ticker = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        report_path = discover_latest_ticker_report(ticker, project_dir)

        if report_path is None:
            return {
                "historical_summary": (
                    f"No historical backtest report was found for {ticker}. "
                    "Proceed with the current debate using live analyst inputs only."
                )
            }

        report_text = report_path.read_text(encoding="utf-8")
        prompt = f"""You are the History Agent.

Your job is to read the backtest report for {ticker} and write a concise conclusion about how the ticker changed over time during the reported period.

Focus on the path of performance, major trend shifts, drawdown severity, leverage or margin pressure, win/loss balance, and what the report implies for the next debate.

Return one short paragraph that the researchers can use as initial context.

Ticker: {ticker}
Asset type: {asset_type}
Report file: {report_path}

<report>
{report_text}
</report>
""" + get_language_instruction()

        response = llm.invoke(prompt)
        summary = getattr(response, "content", "").strip()
        if not summary:
            summary = (
                f"Historical report for {ticker} is available at {report_path.name}, "
                "but no concise conclusion could be generated."
            )

        return {"historical_summary": summary}

    return history_node
