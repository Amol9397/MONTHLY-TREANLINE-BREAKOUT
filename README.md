# NSE Monthly Trend Lab

This version is specifically designed around the requested architecture:

- NSE-listed EQ securities as the universe.
- Monthly-candle strategy scanner.
- No intraday or 4H data in the Monthly Trend Line Break tab.
- Historical bootstrap is aggregated to monthly OHLC.
- Cached historical data is intended to be reused; the scanner should refresh only the new monthly period.
- Filters: descending monthly trendline breakout, Monthly EMA 144, and price above/around 0.50 Fibonacci using all-time high -> all-time low.

NSE official sources:
https://www.nseindia.com/all-reports
https://www.nseindia.com/historical/price-and-volume-data-per-security
