# NSE Monthly Trend-Line Break Scanner — Weekly Swing Version

## What changed

- NSE EQ universe; no sample/fake stocks.
- Monthly candles remain the signal timeframe.
- Descending resistance is now built from **weekly swing highs**, not monthly swing highs.
- Requires two descending weekly swing highs separated by a configurable minimum number of weeks.
- Separate:
  - Confirmed breakout = completed monthly candle
  - In Progress = current monthly candle
- Added minimum 20% fall from all-time high filter.
- Monthly EMA 144.
- ATH → ATL Fibonacci with adjustable 0.50 tolerance.
- Full confirmed/in-progress tables.
- CSV export for both lists.
- Click a qualifying symbol to inspect a monthly candlestick chart with EMA 144, Fib 0.50, ATH and weekly-swing trendline.
- No fallback/sample symbols when NSE data fails.

## Replit run command

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 3000
```

## Important

NSE can rate-limit automated cloud requests. The application intentionally does not replace failed NSE data with fake/sample stocks.
