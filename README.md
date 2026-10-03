# NSE Monthly Trend-Line Break — Weekly Swing v3

Light green + sky-blue + red high-contrast UI. NSE EQ only, monthly signal, weekly swing-high trendline, 20%+ ATH drawdown, EMA 144, ATH→ATL Fib 0.50 tolerance, confirmed/in-progress lists, charts and CSV exports.

### Persistent NSE cache
The app stores each symbol's downloaded NSE history under `.nse_cache/`. On the first scan it downloads the requested history. On later scans it reuses the saved history and requests only a small recent overlap, then merges the new data. The NSE EQ master list is cached too. No fake/sample data is used when NSE fails.

Keep `app.py`, `requirements.txt`, and `README.md` in the GitHub repo. Streamlit Cloud runs `app.py` automatically.
