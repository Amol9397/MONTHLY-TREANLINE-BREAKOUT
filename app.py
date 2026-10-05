import os
import io
import json
import math
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# NSE MONTHLY TREND-LINE BREAKOUT
# Strategy:
# - NSE listed EQ universe
# - Monthly signal timeframe
# - Descending resistance built from WEEKLY swing highs
# - Monthly breakout confirmation
# - Monthly EMA 144
# - ATH -> ATL Fibonacci, 0 at ATH / 1 at ATL
# - Price around/above 0.50 Fib
# - Minimum 20% fall from ATH
# - Confirmed and In Progress signals
# - Local NSE cache to reduce repeat downloads
# ============================================================

st.set_page_config(
    page_title="NSE Monthly Trend Line Breakout",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CACHE_DIR = ".nse_cache"
MASTER_CACHE = os.path.join(CACHE_DIR, "equity_master.csv")
META_CACHE = os.path.join(CACHE_DIR, "cache_meta.json")
os.makedirs(CACHE_DIR, exist_ok=True)

NSE_HOME = "https://www.nseindia.com"
MASTER_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
HIST_URL = "https://www.nseindia.com/api/historical/cm/equity"

# -----------------------------
# High-contrast light theme
# -----------------------------
st.markdown(
    """
<style>
.stApp {
    background: #f3fff7;
    color: #10251a;
}
[data-testid="stHeader"] {
    background: #dff4ff;
}
[data-testid="stHeader"]::after {
    content: "📈 NSE Monthly Trend Line Breakout";
    position: absolute;
    left: 4.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 1.65rem;
    font-weight: 700;
    color: #063b2a;
    white-space: nowrap;
}
[data-testid="stSidebar"] {
    background: #e6fff0;
}
[data-testid="stSidebar"] * {
    color: #10251a !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #063b2a !important;
}
p, label, span, div {
    color: #10251a;
}
.stButton > button {
    background: #e53935;
    color: white !important;
    border: 1px solid #b71c1c;
    border-radius: 8px;
    font-weight: 700;
}
.stButton > button:hover {
    background: #c62828;
    color: white !important;
}
div[data-testid="stMetric"] {
    background: #dff4ff;
    border: 1px solid #8ed1ee;
    padding: 10px;
    border-radius: 10px;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
    color: #063b2a !important;
}
.stDataFrame {
    background: white;
}
div[data-baseweb="select"] > div {
    background: white;
}
.stDownloadButton > button {
    background: #55bfe8;
    color: #052a3a !important;
    font-weight: 700;
    border-radius: 8px;
}
.info-box {
    background: #e0f7ed;
    border-left: 5px solid #159957;
    padding: 12px 15px;
    border-radius: 7px;
    margin: 8px 0;
}
.warning-box {
    background: #fff4d6;
    border-left: 5px solid #e53935;
    padding: 12px 15px;
    border-radius: 7px;
    margin: 8px 0;
}
.small-note {
    font-size: 0.86rem;
    color: #315044 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Session / NSE HTTP
# -----------------------------
@st.cache_resource
def get_nse_session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_HOME + "/",
            "Connection": "keep-alive",
        }
    )
    try:
        s.get(NSE_HOME, timeout=20)
    except Exception:
        pass
    return s


def clean_symbol(symbol):
    return str(symbol).strip().upper().replace("&", "%26").replace(" ", "%20")


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def normalize_ohlc(df):
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()
    x.columns = [str(c).strip().upper() for c in x.columns]

    rename = {}
    for c in x.columns:
        if c in ("OPEN", "OPEN PRICE"):
            rename[c] = "OPEN"
        elif c in ("HIGH", "HIGH PRICE"):
            rename[c] = "HIGH"
        elif c in ("LOW", "LOW PRICE"):
            rename[c] = "LOW"
        elif c in ("CLOSE", "CLOSE PRICE", "CLOSE_PRICE"):
            rename[c] = "CLOSE"
        elif c in ("DATE", "TIMESTAMP", "DATE OF TRADE"):
            rename[c] = "DATE"

    x = x.rename(columns=rename)

    if "DATE" not in x.columns:
        # NSE historical JSON commonly uses CH_TIMESTAMP
        for c in x.columns:
            if "TIMESTAMP" in c:
                x["DATE"] = x[c]
                break

    needed = {"DATE", "OPEN", "HIGH", "LOW", "CLOSE"}
    if not needed.issubset(x.columns):
        return pd.DataFrame()

    x["DATE"] = pd.to_datetime(x["DATE"], errors="coerce", dayfirst=True)
    for c in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        x[c] = pd.to_numeric(
            x[c].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    x = x.dropna(subset=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])
    x = x[["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]]
    x = x.drop_duplicates(subset=["DATE"]).sort_values("DATE")
    x = x.set_index("DATE")
    return x


def fetch_nse_daily(symbol, from_date, to_date, retries=3):
    """Fetch one NSE historical daily window."""
    session = get_nse_session()
    params = {
        "symbol": symbol,
        "from": from_date.strftime("%d-%m-%Y"),
        "to": to_date.strftime("%d-%m-%Y"),
    }

    for attempt in range(retries):
        try:
            r = session.get(HIST_URL, params=params, timeout=30)
            if r.status_code == 200:
                payload = r.json()
                rows = payload.get("data", []) if isinstance(payload, dict) else []
                if rows:
                    raw = pd.DataFrame(rows)
                    # NSE commonly returns fields such as CH_TIMESTAMP,
                    # CH_OPENING_PRICE, CH_TRADE_HIGH_PRICE, etc.
                    rename = {
                        "CH_TIMESTAMP": "DATE",
                        "CH_OPENING_PRICE": "OPEN",
                        "CH_TRADE_HIGH_PRICE": "HIGH",
                        "CH_TRADE_LOW_PRICE": "LOW",
                        "CH_CLOSING_PRICE": "CLOSE",
                    }
                    raw = raw.rename(columns=rename)
                    out = normalize_ohlc(raw)
                    if not out.empty:
                        return out
                return pd.DataFrame()
        except Exception:
            pass

        time.sleep(1.5 * (attempt + 1))
        try:
            session.get(NSE_HOME, timeout=15)
        except Exception:
            pass

    return pd.DataFrame()


# -----------------------------
# NSE master cache
# -----------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_nse_master():
    # First use local cache
    if os.path.exists(MASTER_CACHE):
        try:
            df = pd.read_csv(MASTER_CACHE)
            if not df.empty and "SYMBOL" in df.columns:
                return df
        except Exception:
            pass

    try:
        r = requests.get(
            MASTER_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [str(c).strip().upper() for c in df.columns]

        # Keep NSE EQ securities only.
        if "SERIES" in df.columns:
            df = df[df["SERIES"].astype(str).str.upper().eq("EQ")].copy()

        df = df.dropna(subset=["SYMBOL"]).copy()
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
        df = df.drop_duplicates("SYMBOL").sort_values("SYMBOL")

        df.to_csv(MASTER_CACHE, index=False)
        return df
    except Exception as e:
        if os.path.exists(MASTER_CACHE):
            try:
                return pd.read_csv(MASTER_CACHE)
            except Exception:
                pass
        raise RuntimeError(
            "Unable to load the NSE listed-equity master. "
            f"NSE response/error: {e}"
        )


# -----------------------------
# Per-symbol persistent cache
# -----------------------------
def symbol_cache_path(symbol):
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in symbol)
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def read_symbol_cache(symbol):
    path = symbol_cache_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["DATE"])
        if "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.set_index("DATE")
        return normalize_ohlc(df.reset_index())
    except Exception:
        return pd.DataFrame()


def write_symbol_cache(symbol, df):
    if df is None or df.empty:
        return
    path = symbol_cache_path(symbol)
    out = normalize_ohlc(df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df)
    if out.empty:
        return
    out.to_csv(path)


def nse_history_cached(symbol, start_date, end_date):
    """
    Persistent cache:
    - First run: downloads requested historical period.
    - Later runs: downloads only recent overlap after the cached last date.
    - Merges and de-duplicates local data.
    """
    cached = read_symbol_cache(symbol)

    if cached.empty:
        fresh = fetch_nse_daily(symbol, start_date, end_date)
        if fresh.empty:
            return pd.DataFrame()
        write_symbol_cache(symbol, fresh)
        return fresh

    cached_start = cached.index.min().date()
    cached_end = cached.index.max().date()

    pieces = [cached]

    # Fill missing history at the beginning if the user asks for an earlier range.
    if start_date < cached_start:
        end_before = cached_start - timedelta(days=1)
        before = fetch_nse_daily(symbol, start_date, end_before)
        if not before.empty:
            pieces.append(before)

    # Only fetch recent/missing data after the last cached date.
    if end_date > cached_end:
        fetch_from = max(
            cached_end - timedelta(days=7),
            start_date,
        )
        after = fetch_nse_daily(symbol, fetch_from, end_date)
        if not after.empty:
            pieces.append(after)

    merged = pd.concat(pieces, axis=0)
    merged = normalize_ohlc(merged.reset_index())
    if merged.empty:
        return pd.DataFrame()

    merged = merged.loc[
        (merged.index.date >= start_date)
        & (merged.index.date <= end_date)
    ]

    write_symbol_cache(symbol, merged)
    return merged


def cached_stock_count():
    try:
        return len(
            [
                f
                for f in os.listdir(CACHE_DIR)
                if f.lower().endswith(".csv") and f != "equity_master.csv"
            ]
        )
    except Exception:
        return 0


def clear_cache():
    if not os.path.exists(CACHE_DIR):
        return
    for name in os.listdir(CACHE_DIR):
        path = os.path.join(CACHE_DIR, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass


# -----------------------------
# Timeframe calculations
# -----------------------------
def monthly_from_daily(df):
    if df.empty:
        return pd.DataFrame()

    m = df.resample("ME").agg(
        OPEN=("OPEN", "first"),
        HIGH=("HIGH", "max"),
        LOW=("LOW", "min"),
        CLOSE=("CLOSE", "last"),
    )
    return m.dropna()


def weekly_from_daily(df):
    if df.empty:
        return pd.DataFrame()

    w = df.resample("W-FRI").agg(
        OPEN=("OPEN", "first"),
        HIGH=("HIGH", "max"),
        LOW=("LOW", "min"),
        CLOSE=("CLOSE", "last"),
    )
    return w.dropna()


def ema(series, period=144):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def swing_high_indices(highs, left=2, right=2):
    vals = highs.reset_index(drop=True)
    indices = []
    for i in range(left, len(vals) - right):
        window = vals.iloc[i - left : i + right + 1]
        if vals.iloc[i] >= window.max():
            indices.append(i)
    return indices


def descending_weekly_trendline(
    weekly,
    left=2,
    right=2,
    min_gap_weeks=4,
    max_gap_weeks=104,
):
    """
    Return two weekly swing highs defining a descending resistance line.

    The line is calculated in weekly index-space. It is intentionally
    based on WEEKLY swing highs, while breakout is evaluated on MONTHLY bars.
    """
    if weekly.empty or len(weekly) < max(20, left + right + 5):
        return None

    highs = weekly["HIGH"]
    idxs = swing_high_indices(highs, left, right)

    if len(idxs) < 2:
        return None

    # Prefer the latest confirmed swing high as the second pivot.
    for j in range(len(idxs) - 1, 0, -1):
        i2 = idxs[j]

        for k in range(j - 1, -1, -1):
            i1 = idxs[k]
            gap = i2 - i1

            if gap < min_gap_weeks:
                continue
            if gap > max_gap_weeks:
                continue

            y1 = float(highs.iloc[i1])
            y2 = float(highs.iloc[i2])

            # Descending resistance only.
            if y2 >= y1:
                continue

            slope = (y2 - y1) / float(i2 - i1)
            intercept = y1 - slope * i1

            return {
                "i1": i1,
                "i2": i2,
                "d1": weekly.index[i1],
                "d2": weekly.index[i2],
                "p1": y1,
                "p2": y2,
                "slope": slope,
                "intercept": intercept,
            }

    return None


def trendline_value(line, weekly_index_float):
    if line is None:
        return np.nan
    return line["intercept"] + line["slope"] * weekly_index_float


def weekly_position_for_date(weekly, dt):
    if weekly.empty:
        return np.nan
    ts = pd.Timestamp(dt)
    pos = weekly.index.searchsorted(ts, side="right") - 1
    pos = max(0, min(pos, len(weekly) - 1))
    return float(pos)


def fib_0_5(ath, atl):
    return float(ath - 0.5 * (ath - atl))


def evaluate_symbol(
    symbol,
    daily,
    fib_tolerance_pct=5.0,
    min_fall_pct=20.0,
    swing_left=2,
    swing_right=2,
    min_gap_weeks=4,
    max_gap_weeks=104,
    breakout_buffer_pct=0.0,
):
    """
    Returns one result dict or None.

    Important:
    - ATH/ATL are calculated from available requested historical data.
    - Monthly EMA144 is calculated on monthly candles.
    - Weekly swing-high trendline is used as monthly resistance.
    - Current month is explicitly detected from today's date.
    """
    if daily.empty:
        return None

    daily = normalize_ohlc(daily.reset_index())
    if daily.empty or len(daily) < 300:
        return None

    monthly = monthly_from_daily(daily)
    weekly = weekly_from_daily(daily)

    if len(monthly) < 150 or len(weekly) < 30:
        return None

    monthly["EMA144"] = ema(monthly["CLOSE"], 144)

    line = descending_weekly_trendline(
        weekly,
        left=swing_left,
        right=swing_right,
        min_gap_weeks=min_gap_weeks,
        max_gap_weeks=max_gap_weeks,
    )
    if line is None:
        return None

    # Determine whether the latest monthly candle is still forming.
    today = pd.Timestamp.today().normalize()
    current_month_start = today.to_period("M").start_time
    current_month_end = today.to_period("M").end_time.normalize()

    latest_month = monthly.index[-1]
    latest_month_is_current = (
        latest_month.to_period("M") == current_month_start.to_period("M")
        and today < current_month_end
    )

    if latest_month_is_current:
        current = monthly.iloc[-1]
        completed = monthly.iloc[:-1]
    else:
        current = None
        completed = monthly

    if len(completed) < 2:
        return None

    # ATH / ATL from the complete available history.
    ath = float(daily["HIGH"].max())
    atl = float(daily["LOW"].min())

    if not np.isfinite(ath) or not np.isfinite(atl) or ath <= atl:
        return None

    fib50 = fib_0_5(ath, atl)
    fib_lower = fib50 * (1.0 - fib_tolerance_pct / 100.0)

    # 0 at ATH, 1 at ATL. "Above or around 0.50" means
    # price >= the lower edge of the tolerance band.
    def filters(price, ema_value):
        if not np.isfinite(ema_value):
            return False, False, False, False

        fib_ok = price >= fib_lower
        ema_ok = price > ema_value
        fall_pct = (ath - price) / ath * 100.0
        fall_ok = fall_pct >= min_fall_pct
        return fib_ok and ema_ok and fall_ok, fib_ok, ema_ok, fall_ok

    # Evaluate latest completed month for confirmed breakout.
    c = completed.iloc[-1]
    p = completed.iloc[-2]

    c_pos = weekly_position_for_date(weekly, completed.index[-1])
    p_pos = weekly_position_for_date(weekly, completed.index[-2])

    c_line = trendline_value(line, c_pos)
    p_line = trendline_value(line, p_pos)

    c_price = float(c["CLOSE"])
    p_price = float(p["CLOSE"])

    passes, fib_ok, ema_ok, fall_ok = filters(
        c_price,
        float(c["EMA144"]),
    )

    breakout = (
        p_price <= p_line
        and c_price > c_line * (1.0 + breakout_buffer_pct / 100.0)
    )

    confirmed = bool(breakout and passes)

    fall_pct = (ath - c_price) / ath * 100.0
    fib_level = (ath - c_price) / (ath - atl) if ath != atl else np.nan

    base = {
        "SYMBOL": symbol,
        "PRICE": round(c_price, 2),
        "EMA144": round(float(c["EMA144"]), 2),
        "ATH": round(ath, 2),
        "ATL": round(atl, 2),
        "FIB_0.50": round(fib50, 2),
        "FIB_LEVEL": round(float(fib_level), 3),
        "FALL_FROM_ATH_%": round(fall_pct, 2),
        "TRENDLINE": round(float(c_line), 2),
        "BREAKOUT_DATE": completed.index[-1].strftime("%Y-%m-%d"),
        "SWING_HIGH_1": line["d1"].strftime("%Y-%m-%d"),
        "SWING_HIGH_1_PRICE": round(line["p1"], 2),
        "SWING_HIGH_2": line["d2"].strftime("%Y-%m-%d"),
        "SWING_HIGH_2_PRICE": round(line["p2"], 2),
    }

    if confirmed:
        base["STATUS"] = "CONFIRMED"
        return base

    # In-progress current month:
    # Only show if the current month's price has crossed the projected
    # weekly-swing-high resistance while the previous completed month
    # remained below/equal to the line.
    if current is not None:
        cur_pos = weekly_position_for_date(weekly, monthly.index[-1])
        cur_line = trendline_value(line, cur_pos)

        cur_price = float(current["CLOSE"])
        cur_ema = float(current["EMA144"])

        cur_passes, cur_fib, cur_ema_ok, cur_fall_ok = filters(
            cur_price,
            cur_ema,
        )

        in_progress_break = (
            p_price <= p_line
            and cur_price > cur_line * (1.0 + breakout_buffer_pct / 100.0)
        )

        if in_progress_break and cur_passes:
            cur_fall = (ath - cur_price) / ath * 100.0
            cur_fib_level = (
                (ath - cur_price) / (ath - atl)
                if ath != atl
                else np.nan
            )

            return {
                **base,
                "STATUS": "IN PROGRESS",
                "PRICE": round(cur_price, 2),
                "EMA144": round(cur_ema, 2),
                "TRENDLINE": round(float(cur_line), 2),
                "FALL_FROM_ATH_%": round(cur_fall, 2),
                "FIB_LEVEL": round(float(cur_fib_level), 3),
                "BREAKOUT_DATE": today.strftime("%Y-%m-%d"),
            }

    return None


# -----------------------------
# Scanner
# -----------------------------
def scan_universe(
    symbols,
    start_date,
    end_date,
    fib_tolerance_pct,
    min_fall_pct,
    swing_left,
    swing_right,
    min_gap_weeks,
    max_gap_weeks,
    breakout_buffer_pct,
):
    results = []
    errors = 0
    progress = st.progress(0, text="Preparing NSE scan...")

    total = len(symbols)

    for n, symbol in enumerate(symbols, start=1):
        try:
            daily = nse_history_cached(
                symbol,
                start_date,
                end_date,
            )
            if not daily.empty:
                result = evaluate_symbol(
                    symbol,
                    daily,
                    fib_tolerance_pct=fib_tolerance_pct,
                    min_fall_pct=min_fall_pct,
                    swing_left=swing_left,
                    swing_right=swing_right,
                    min_gap_weeks=min_gap_weeks,
                    max_gap_weeks=max_gap_weeks,
                    breakout_buffer_pct=breakout_buffer_pct,
                )
                if result:
                    results.append(result)
        except Exception:
            errors += 1

        progress.progress(
            n / total,
            text=f"Scanning NSE EQ: {n}/{total} — {symbol}",
        )

    progress.empty()

    if not results:
        return pd.DataFrame(), errors

    df = pd.DataFrame(results)
    status_order = {"CONFIRMED": 0, "IN PROGRESS": 1}
    df["_ORDER"] = df["STATUS"].map(status_order).fillna(9)
    df = df.sort_values(
        ["_ORDER", "FALL_FROM_ATH_%"],
        ascending=[True, False],
    ).drop(columns=["_ORDER"])

    return df.reset_index(drop=True), errors


# -----------------------------
# Chart
# -----------------------------
def make_monthly_chart(symbol, daily, result=None):
    monthly = monthly_from_daily(daily)
    weekly = weekly_from_daily(daily)

    if monthly.empty:
        return go.Figure()

    monthly["EMA144"] = ema(monthly["CLOSE"], 144)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=monthly.index,
            open=monthly["OPEN"],
            high=monthly["HIGH"],
            low=monthly["LOW"],
            close=monthly["CLOSE"],
            name="Monthly",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["EMA144"],
            mode="lines",
            name="EMA 144",
            line=dict(width=2),
        )
    )

    if result is not None:
        try:
            # Rebuild the weekly resistance line from the displayed data.
            line = descending_weekly_trendline(weekly)
            if line:
                x0 = monthly.index.min()
                x1 = monthly.index.max()

                p0 = weekly_position_for_date(weekly, x0)
                p1 = weekly_position_for_date(weekly, x1)

                y0 = trendline_value(line, p0)
                y1 = trendline_value(line, p1)

                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        name="Weekly Swing-High Resistance",
                        line=dict(width=3, dash="dash"),
                    )
                )

            ath = float(result["ATH"])
            atl = float(result["ATL"])
            fib50 = fib_0_5(ath, atl)

            fig.add_hline(
                y=fib50,
                line_dash="dot",
                annotation_text="Fib 0.50",
            )

        except Exception:
            pass

    fig.update_layout(
        title=f"{symbol} — Monthly Chart",
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#10251a"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


# ============================================================
# UI
# ============================================================
st.markdown(
    '<div class="info-box">'
    "<b>Monthly signal + Weekly swing-high resistance</b><br>"
    "NSE-listed EQ universe • ATH→ATL Fibonacci • EMA 144 • minimum 20% ATH fall"
    "</div>",
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.header("Scanner Settings")

    years = st.slider(
        "Historical data (years)",
        min_value=3,
        max_value=15,
        value=8,
        step=1,
    )

    fib_tolerance = st.slider(
        "Fib 0.50 tolerance (%)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
    )

    min_fall = st.slider(
        "Minimum fall from ATH (%)",
        min_value=20.0,
        max_value=80.0,
        value=20.0,
        step=5.0,
    )

    swing_left = st.number_input(
        "Weekly swing-high left bars",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    swing_right = st.number_input(
        "Weekly swing-high right bars",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    min_gap = st.number_input(
        "Minimum swing-high gap (weeks)",
        min_value=2,
        max_value=52,
        value=4,
        step=1,
    )

    max_gap = st.number_input(
        "Maximum swing-high gap (weeks)",
        min_value=10,
        max_value=260,
        value=104,
        step=1,
    )

    breakout_buffer = st.number_input(
        "Breakout buffer (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
    )

    st.divider()

    if st.button("🧹 Clear local NSE cache", use_container_width=True):
        clear_cache()
        st.cache_data.clear()
        st.success("Local cache cleared. Next scan will rebuild required NSE data.")
        st.rerun()

    st.caption(f"Cached stock files: {cached_stock_count()}")

# Load master
try:
    master = get_nse_master()
except Exception as e:
    st.error(str(e))
    st.stop()

symbols = master["SYMBOL"].dropna().astype(str).str.upper().drop_duplicates().tolist()

c1, c2, c3, c4 = st.columns(4)
c1.metric("NSE EQ universe", f"{len(symbols):,}")
c2.metric("Cached stocks", f"{cached_stock_count():,}")
c3.metric("Signal timeframe", "Monthly")
c4.metric("Trendline source", "Weekly")

st.markdown(
    '<div class="small-note">'
    "First scan may take time because historical NSE data must be bootstrapped. "
    "Afterward the local cache is reused and only missing/recent NSE data is requested."
    "</div>",
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([1, 3])

with col_a:
    scan = st.button(
        "🔴 SCAN NSE STOCKS",
        use_container_width=True,
    )

with col_b:
    st.write(
        f"Data window: {date.today() - timedelta(days=365 * years)} "
        f"to {date.today()}"
    )

if scan:
    start = date.today() - timedelta(days=365 * years)
    end = date.today()

    with st.spinner("Scanning NSE-listed EQ securities..."):
        result_df, error_count = scan_universe(
            symbols=symbols,
            start_date=start,
            end_date=end,
            fib_tolerance_pct=fib_tolerance,
            min_fall_pct=min_fall,
            swing_left=int(swing_left),
            swing_right=int(swing_right),
            min_gap_weeks=int(min_gap),
            max_gap_weeks=int(max_gap),
            breakout_buffer_pct=breakout_buffer,
        )

    st.session_state["scan_results"] = result_df
    st.session_state["scan_errors"] = error_count

# Results
result_df = st.session_state.get("scan_results", pd.DataFrame())
error_count = st.session_state.get("scan_errors", 0)

if result_df.empty:
    st.markdown(
        '<div class="warning-box">'
        "<b>No scan results loaded.</b><br>"
        "Click <b>SCAN NSE STOCKS</b> to run the monthly breakout scan. "
        "No sample/fake stocks are displayed."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    confirmed = result_df[result_df["STATUS"].eq("CONFIRMED")].copy()
    in_progress = result_df[result_df["STATUS"].eq("IN PROGRESS")].copy()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total qualifying", len(result_df))
    m2.metric("Confirmed", len(confirmed))
    m3.metric("In Progress", len(in_progress))

    if error_count:
        st.caption(
            f"{error_count} symbols returned no usable data or an NSE request error."
        )

    display_cols = [
        "SYMBOL",
        "STATUS",
        "PRICE",
        "EMA144",
        "ATH",
        "ATL",
        "FIB_0.50",
        "FIB_LEVEL",
        "FALL_FROM_ATH_%",
        "TRENDLINE",
        "BREAKOUT_DATE",
        "SWING_HIGH_1",
        "SWING_HIGH_1_PRICE",
        "SWING_HIGH_2",
        "SWING_HIGH_2_PRICE",
    ]
    display_cols = [c for c in display_cols if c in result_df.columns]

    st.subheader("Breakout List")
    st.dataframe(
        result_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Breakout CSV",
        data=csv_bytes,
        file_name="nse_monthly_trendline_breakout.csv",
        mime="text/csv",
    )

    st.divider()

    available_symbols = result_df["SYMBOL"].tolist()
    selected = st.selectbox(
        "Open stock chart",
        available_symbols,
    )

    # Load cached history for selected stock.
    selected_daily = nse_history_cached(
        selected,
        date.today() - timedelta(days=365 * years),
        date.today(),
    )

    selected_result = result_df[
        result_df["SYMBOL"].eq(selected)
    ].iloc[0].to_dict()

    if not selected_daily.empty:
        st.plotly_chart(
            make_monthly_chart(
                selected,
                selected_daily,
                selected_result,
            ),
            use_container_width=True,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Status", selected_result.get("STATUS", "-"))
        r2.metric("Price", selected_result.get("PRICE", "-"))
        r3.metric("EMA 144", selected_result.get("EMA144", "-"))
        r4.metric("Fall from ATH", f'{selected_result.get("FALL_FROM_ATH_%", "-")}%')

# Footer
st.divider()
st.caption(
    "Research/educational scanner. NSE data availability and cloud/network "
    "limits can affect scan completeness. No profit guarantee."
)
