from __future__ import annotations

from pathlib import Path
from typing import Optional

from tradingagents.dataflows.utils import safe_ticker_component


def _path_matches_ticker(path: Path, ticker: str) -> bool:
    return any(part == ticker or part.startswith(f"{ticker}_") for part in path.parts)


def discover_latest_ticker_report(ticker: str, project_dir: str) -> Optional[Path]:
    """Find the newest report.md for a ticker in the local workspace."""
    safe_ticker = safe_ticker_component(ticker)
    project_path = Path(project_dir)
    candidate_paths: list[Path] = []

    for root_name in ("backtest_results", "reports"):
        root = project_path / root_name
        if not root.exists():
            continue

        exact = root / safe_ticker / "report.md"
        if exact.exists():
            candidate_paths.append(exact)

        for report_path in root.rglob("report.md"):
            if _path_matches_ticker(report_path, safe_ticker):
                candidate_paths.append(report_path)

    unique_paths = []
    seen = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    if not unique_paths:
        return None

    return max(unique_paths, key=lambda path: (path.stat().st_mtime, len(path.parts), str(path)))
