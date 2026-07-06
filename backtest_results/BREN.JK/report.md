# Backtest Report - BREN.JK (Stock)

## Period

- Asset Class: `stock`
- Ticker: `BREN.JK`
- Start Date: `2026-05-25`
- End Date: `2026-06-30`
- Initial Cash: `100,000,000.00`
- Initial Margin %: `50.00%`
- Maintenance Margin %: `35.00%`

## Performance Summary

- Final Equity: `89,757,748.80`
- Total Return: `-10.24%`
- CAGR: `-66.59%`
- Max Drawdown: `-52.07%`
- Sharpe Ratio: `1.2071`
- Sortino Ratio: `1.9140`
- Win Rate: `40.00%`
- Profit Factor: `0.4013290801864925`
- Number of Trades: `10`
- Exposure Time: `76.00%`
- Benchmark Return: `None`
- Alpha: `None`

## Margin-Specific Metrics

- Average Leverage: `0.24x`
- Max Leverage: `0.45x`
- Average Margin Utilization: `11.29%`
- Max Margin Utilization: `21.74%`
- Margin Calls: `0`
- Liquidations: `0`
- Reverse Count: `0`
- Avg Holding Period (days): `0.00`
- Max Consecutive Wins: `2`
- Max Consecutive Losses: `3`
- Long Win Rate: `50.00%`
- Short Win Rate: `33.33%`
- Long Realized PnL: `-1,556,856.00`
- Short Realized PnL: `-7,907,485.87`
- Calmar Ratio: `-127.8906`
- Fee Drag: `0.7779%`
- Slippage Drag: `0.0000%`
- Total Fees: `777,909.33`
- Total Slippage: `0.00`
- Turnover Notional: `404,210,063.88`

## Margin Activity

| Date | Kind | Mark | Deficit | Account Equity | Action |
|------|------|------|---------|----------------|--------|
| 2026-05-29 | hard_risk | 2337.66 | 0.00 | 117,255,395.71 | Max single trade risk exceeded: 10.34% |
| 2026-06-05 | hard_risk | 4154.15 | 0.00 | 60,251,918.15 | Max single trade risk exceeded: 5.90% |
| 2026-06-30 | take_profit | 3097.68 | 0.00 | 114,156,987.47 | Short target triggered: low 3040.0 <= target 3097.6785714285716 |

## Leakage Audit

Status: **PASSED**

- `memory_isolation`: **PASSED**
- `live_provider_disabled`: **PASSED**
- `lookback_window_respected`: **PASSED**
- `required_decision_fields`: **PASSED**
- `snapshot_metadata_required`: **PASSED**
- `ohlcv_cutoff`: **PASSED**
- `last_data_date_cutoff`: **PASSED**
- `news_cutoff`: **PASSED**
- `fundamental_available_date`: **PASSED**
- `sentiment_cutoff`: **PASSED**
- `next_bar_execution`: **PASSED**

## Notes

- Asset class: **stock**. Margin, long+short, daily mark-to-market settlement, auto-liquidation.
- Decision pada tanggal `t` hanya dieksekusi pada trading day berikutnya.
- Harga eksekusi memakai `next session open` dengan ``tick_slippage`` ticks.
- Live provider wajib disabled saat backtest historis.
- ``account_state.csv`` berisi snapshot harian (cash, position, margin, leverage).
