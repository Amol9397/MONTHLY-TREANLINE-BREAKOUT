
import streamlit as st
import pandas as pd
import numpy as np
import requests, io, time
from datetime import date, timedelta

st.set_page_config(
    page_title="NSE Monthly Breakout Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: #07111f;
    color: #f4f8ff;
}
section[data-testid="stSidebar"] {
    background: #0a1728;
    border-right: 1px solid #24415f;
}
h1,h2,h3 { color: #f4f8ff !important; }
.stCaption { color: #b9c8d8 !important; }
div[data-testid="stMetric"] {
    background: #0d2035;
    border: 1px solid #2b5578;
    border-radius: 14px;
    padding: 10px;
}
div.stButton > button {
    border-radius: 10px;
    font-weight: 800;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #284b6b;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 NSE MONTHLY BREAKOUT SCANNER")
st.caption("NSE EQ universe • monthly signal • weekly swing structure • no sample/fake symbols")

# -----------------------------
# Settings
# -----------------------------
st.sidebar.header("Scanner settings")

fib_tol = st.sidebar.slider(
    "0.50 Fib 'around' tolerance (%)", 0, 15, 5, 1
)
break_buffer = st.sidebar.slider(
    "Breakout buffer (%)", 0.0, 5.0, 0.5, 0.1
)
ath_fall_min = st.sidebar.slider(
    "Minimum fall from ATH (%)", 20, 60, 20, 1
)
weekly_left = st.sidebar.slider(
    "Weekly swing left bars", 1, 5, 2, 1
)
weekly_right = st.sidebar.slider(
    "Weekly swing right bars", 1, 5, 2, 1
)
min_weekly_gap = st.sidebar.slider(
    "Minimum weekly swing-high separation", 2, 20, 4, 1
)
min_history = st.sidebar.number_input(
    "Minimum completed monthly candles", 150, 500, 160, 10
)

# -----------------------------
# NSE
# -----------------------------
NSE_MASTER = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_HOME = "https://www.nseindia.com"
NSE_HIST = "https://www.nseindia.com/api/historical/cm/equity"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": NSE_HOME + "/",
    "Connection": "keep-alive",
}

def nse_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(NSE_HOME, timeout=20)
    return s

@st.cache_data(ttl=86400, show_spinner=False)
def nse_equity_master():
    s = nse_session()
    r = s.get(NSE_MASTER, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [str(c).strip() for c in df.columns]
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.upper().eq("EQ")].copy()
    if "SYMBOL" in df.columns:
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    return df.reset_index(drop=True)

@st.cache_data(ttl=2592000, show_spinner=False)
def nse_daily_history(symbol, start, end):
    s = nse_session()
    params = {
        "symbol": symbol,
        "series": '["EQ"]',
        "from": start,
        "to": end,
    }
    r = s.get(NSE_HIST, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    rows = j.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename = {}
    for c in df.columns:
        u = str(c).upper()
        if u == "CH_CLOSING_PRICE": rename[c] = "CLOSE"
        elif u == "CH_OPENING_PRICE": rename[c] = "OPEN"
        elif u == "CH_TRADE_HIGH_PRICE": rename[c] = "HIGH"
        elif u == "CH_TRADE_LOW_PRICE": rename[c] = "LOW"
        elif u == "CH_TIMESTAMP": rename[c] = "DATE"

    df = df.rename(columns=rename)
    required = {"DATE", "OPEN", "HIGH", "LOW", "CLOSE"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    for c in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])
    return df.sort_values("DATE").drop_duplicates("DATE")

def to_monthly(df):
    if df.empty:
        return df
    x = df.set_index("DATE")[["OPEN", "HIGH", "LOW", "CLOSE"]].resample("ME").agg(
        {"OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last"}
    ).dropna()
    return x

def to_weekly(df):
    if df.empty:
        return df
    x = df.set_index("DATE")[["OPEN", "HIGH", "LOW", "CLOSE"]].resample("W-FRI").agg(
        {"OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last"}
    ).dropna()
    return x

def ema(series, n=144):
    return series.ewm(span=n, adjust=False).mean()

# -----------------------------
# Weekly swing structure
# -----------------------------
def swing_highs(highs, left=2, right=2):
    vals = highs.reset_index(drop=True)
    idx = []
    for i in range(left, len(vals) - right):
        window = vals.iloc[i-left:i+right+1]
        if vals.iloc[i] >= window.max():
            idx.append(i)
    return idx

def swing_lows(lows, left=2, right=2):
    vals = lows.reset_index(drop=True)
    idx = []
    for i in range(left, len(vals) - right):
        window = vals.iloc[i-left:i+right+1]
        if vals.iloc[i] <= window.min():
            idx.append(i)
    return idx

def weekly_descending_trendline(weekly, left, right, min_gap):
    """
    Trendline is based on WEEKLY swing highs, not monthly pivots.
    We select the latest meaningful pair of descending swing highs.
    """
    if len(weekly) < max(30, left + right + 10):
        return None

    hs = swing_highs(weekly["HIGH"], left, right)
    if len(hs) < 2:
        return None

    # Search recent pairs first. Require a minimum bar separation.
    candidates = []
    for a_pos in range(len(hs) - 2, -1, -1):
        for b_pos in range(len(hs) - 1, a_pos, -1):
            i, j = hs[a_pos], hs[b_pos]
            if j - i < min_gap:
                continue
            hi, hj = float(weekly["HIGH"].iloc[i]), float(weekly["HIGH"].iloc[j])
            if hj >= hi:
                continue
            slope = (hj - hi) / (j - i)
            if slope >= 0:
                continue
            t1 = weekly.index[i]
            t2 = weekly.index[j]
            candidates.append((j, i, slope, hi - slope * i, t1, t2))

    if not candidates:
        return None

    # Latest second pivot first; among those, prefer the closest valid first pivot.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, i, slope, intercept, t1, t2 = candidates[0]

    return {
        "slope": slope,
        "intercept": intercept,
        "i": i,
        "j": hs[[x[0] for x in candidates].index(candidates[0][0])] if False else None,
        "pivot1_date": t1,
        "pivot2_date": t2,
    }

def line_value_at_date(weekly, tl, dt):
    if tl is None:
        return np.nan
    # Weekly index is evenly spaced enough for this coordinate projection.
    # Use ordinal week number from the first weekly candle.
    first = weekly.index[0]
    x = (pd.Timestamp(dt) - pd.Timestamp(first)).days / 7.0
    # Recreate intercept in the same weekly-bar coordinate system.
    # Store x coordinates explicitly below where needed.
    return tl["slope"] * x + tl["intercept"]

def build_weekly_trendline(weekly, left, right, min_gap):
    hs = swing_highs(weekly["HIGH"], left, right)
    if len(hs) < 2:
        return None

    candidates = []
    for a in range(len(hs)-2, -1, -1):
        for b in range(len(hs)-1, a, -1):
            i, j = hs[a], hs[b]
            if j - i < min_gap:
                continue
            h1, h2 = float(weekly["HIGH"].iloc[i]), float(weekly["HIGH"].iloc[j])
            if h2 >= h1:
                continue
            slope = (h2 - h1) / (j - i)
            intercept = h1 - slope * i
            candidates.append({
                "i": i, "j": j, "slope": slope, "intercept": intercept,
                "pivot1_date": weekly.index[i], "pivot2_date": weekly.index[j],
                "pivot1_price": h1, "pivot2_price": h2
            })

    if not candidates:
        return None

    # Most recent valid second swing high, then most recent first swing high.
    candidates.sort(key=lambda x: (x["j"], x["i"]), reverse=True)
    return candidates[0]

def weekly_line_at_index(tl, weekly_index):
    return tl["slope"] * weekly_index + tl["intercept"]

def weekly_line_at_date(tl, weekly, dt):
    # Interpolate date onto the weekly index coordinate.
    pos = weekly.index.searchsorted(pd.Timestamp(dt), side="right") - 1
    if pos < 0:
        pos = 0
    return weekly_line_at_index(tl, pos)

# -----------------------------
# Signal engine
# -----------------------------
def evaluate_symbol(raw, fib_tol, break_buffer, ath_fall_min,
                    weekly_left, weekly_right, min_weekly_gap,
                    min_history):
    if raw.empty:
        return None

    monthly = to_monthly(raw)
    weekly = to_weekly(raw)

    if len(monthly) < min_history or len(weekly) < 40:
        return None

    # Completed monthly candles for confirmed signal.
    completed = monthly.iloc[:-1].copy()
    if len(completed) < min_history:
        return None

    # Current month, if it exists, is separately checked as "in progress".
    current_month = monthly.iloc[-1].copy()
    current_month_date = monthly.index[-1]

    # ATH/ATL from the available NSE history.
    ath = float(monthly["HIGH"].max())
    atl = float(monthly["LOW"].min())
    fib50 = ath - (ath - atl) * 0.50

    # Use completed monthly close for the confirmed state.
    close = float(completed["CLOSE"].iloc[-1])
    prev_close = float(completed["CLOSE"].iloc[-2])

    ema144 = float(ema(completed["CLOSE"], 144).iloc[-1])

    # 20%+ below ATH.
    fall_pct = (ath - close) / ath * 100 if ath else 0.0
    fall_ok = fall_pct >= ath_fall_min

    fib_gap_pct = (close / fib50 - 1) * 100 if fib50 else np.nan
    fib_ok = close >= fib50 * (1 - fib_tol / 100)

    # Build weekly structure using data available through the completed month.
    completed_end = completed.index[-1]
    weekly_completed = weekly[weekly.index <= completed_end]
    tl = build_weekly_trendline(
        weekly_completed, weekly_left, weekly_right, min_weekly_gap
    )

    confirmed = False
    in_progress = False
    confirmed_line = np.nan
    progress_line = np.nan
    pivot_text = ""

    if tl is not None:
        confirmed_line = weekly_line_at_date(tl, weekly_completed, completed_end)
        prev_date = completed.index[-2]
        prev_line = weekly_line_at_date(tl, weekly_completed, prev_date)

        # Monthly breakout is confirmed by the completed monthly close.
        confirmed = (
            prev_close <= prev_line and
            close > confirmed_line * (1 + break_buffer / 100)
        )

        pivot_text = (
            f'{pd.Timestamp(tl["pivot1_date"]).strftime("%Y-%m-%d")} '
            f'({tl["pivot1_price"]:.2f}) → '
            f'{pd.Timestamp(tl["pivot2_date"]).strftime("%Y-%m-%d")} '
            f'({tl["pivot2_price"]:.2f})'
        )

    # Current-month "in progress" uses the current monthly candle,
    # but the trendline is still defined from completed weekly swings.
    if tl is not None and current_month_date > completed_end:
        cur_close = float(current_month["CLOSE"])
        cur_line = weekly_line_at_date(tl, weekly, current_month_date)
        progress_line = cur_line

        # Compare the latest completed monthly close with the projected weekly line.
        in_progress = (
            close <= confirmed_line if np.isfinite(confirmed_line) else False
        ) and (
            cur_close > cur_line * (1 + break_buffer / 100)
        )

        cur_ema = float(ema(monthly["CLOSE"].iloc[:-1], 144).iloc[-1])
        cur_fall_pct = (ath - cur_close) / ath * 100 if ath else 0.0
        cur_fib_ok = cur_close >= fib50 * (1 - fib_tol / 100)

        # Apply the 20% ATH fall + EMA + Fib filters to the live candle too.
        in_progress = in_progress and (
            cur_close > cur_ema and
            cur_fib_ok and
            cur_fall_pct >= ath_fall_min
        )

    # Confirmed filters.
    confirmed = confirmed and (close > ema144) and fib_ok and fall_ok

    if not confirmed and not in_progress:
        return None

    status = []
    if confirmed:
        status.append("🟢 Confirmed")
    if in_progress:
        status.append("🟡 In Progress")

    return {
        "Status": " + ".join(status),
        "Close": round(close, 2),
        "ATH": round(ath, 2),
        "Fall from ATH %": round(fall_pct, 2),
        "EMA 144": round(ema144, 2),
        "Fib 0.50": round(fib50, 2),
        "Fib gap %": round(fib_gap_pct, 2),
        "Trendline": round(confirmed_line, 2) if np.isfinite(confirmed_line) else np.nan,
        "Breakout month": completed.index[-1].strftime("%Y-%m"),
        "Weekly swing highs": pivot_text,
        "Trendline slope": round(float(tl["slope"]), 4) if tl else np.nan,
    }

# -----------------------------
# Chart
# -----------------------------
def show_chart(raw, symbol, company, fib_tol, break_buffer,
               ath_fall_min, weekly_left, weekly_right, min_weekly_gap):
    try:
        import plotly.graph_objects as go
    except Exception:
        st.warning("Plotly is not installed.")
        return

    monthly = to_monthly(raw)
    weekly = to_weekly(raw)
    if monthly.empty:
        return

    ath = float(monthly["HIGH"].max())
    atl = float(monthly["LOW"].min())
    fib50 = ath - (ath - atl) * 0.50
    ema144 = ema(monthly["CLOSE"], 144)

    completed = monthly.iloc[:-1]
    tl = build_weekly_trendline(
        weekly[weekly.index <= completed.index[-1]],
        weekly_left, weekly_right, min_weekly_gap
    ) if len(completed) else None

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=monthly.index,
        open=monthly["OPEN"],
        high=monthly["HIGH"],
        low=monthly["LOW"],
        close=monthly["CLOSE"],
        name="Monthly",
    ))
    fig.add_trace(go.Scatter(
        x=monthly.index, y=ema144,
        mode="lines", name="EMA 144"
    ))
    fig.add_hline(y=fib50, line_dash="dot", annotation_text="Fib 0.50")

    if tl:
        xs = [monthly.index[0], monthly.index[-1]]
        ys = [weekly_line_at_date(tl, weekly, x) for x in xs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name="Weekly swing-high trendline"
        ))

        # Mark the two weekly swing highs.
        fig.add_trace(go.Scatter(
            x=[tl["pivot1_date"], tl["pivot2_date"]],
            y=[tl["pivot1_price"], tl["pivot2_price"]],
            mode="markers+text",
            text=["Swing H1", "Swing H2"],
            textposition="top center",
            name="Weekly swing highs"
        ))

    fig.add_hline(y=ath, line_dash="dash", annotation_text="ATH")

    fig.update_layout(
        height=620,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
        title=f"{symbol} — {company or ''} — Monthly structure",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# App
# -----------------------------
tab_dashboard, tab_scanner = st.tabs(["🏠 Dashboard", "🔷 Monthly Breakout Scanner"])

with tab_dashboard:
    st.subheader("NSE Monthly Breakout Scanner")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe", "NSE EQ")
    c2.metric("Signal timeframe", "Monthly")
    c3.metric("Structure", "Weekly swings")
    c4.metric("Direction", "Long / breakout")

with tab_scanner:
    st.subheader("🔷 Monthly Trend-Line Break")
    st.caption(
        "Weekly swing highs define the descending resistance line; "
        "the breakout is confirmed on the monthly candle."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh NSE Universe"):
            st.cache_data.clear()
            st.rerun()

    try:
        master = nse_equity_master()
        symbol_col = "SYMBOL" if "SYMBOL" in master.columns else master.columns[0]
        st.success(f"NSE EQ universe loaded: {len(master):,} securities")
    except Exception as e:
        st.error(f"NSE universe could not be loaded: {e}")
        master = pd.DataFrame()

    if not master.empty:
        if st.button("🚀 Scan NSE Monthly Setups", type="primary"):
            symbols = (
                master[symbol_col]
                .dropna()
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )

            start = (date.today() - timedelta(days=365 * 25)).strftime("%d-%m-%Y")
            end = date.today().strftime("%d-%m-%Y")

            confirmed_rows = []
            progress_rows = []
            failures = 0

            prog = st.progress(0)
            status_box = st.empty()

            company_map = {}
            if "NAME OF COMPANY" in master.columns:
                company_map = dict(zip(
                    master[symbol_col].astype(str).str.strip(),
                    master["NAME OF COMPANY"].astype(str)
                ))

            for k, sym in enumerate(symbols):
                try:
                    raw = nse_daily_history(sym, start, end)
                    result = evaluate_symbol(
                        raw, fib_tol, break_buffer, ath_fall_min,
                        weekly_left, weekly_right, min_weekly_gap,
                        min_history
                    )

                    if result:
                        result["Symbol"] = sym
                        result["Company"] = company_map.get(sym, "")
                        if "🟢 Confirmed" in result["Status"]:
                            confirmed_rows.append(result)
                        if "🟡 In Progress" in result["Status"]:
                            progress_rows.append(result)
                except Exception:
                    failures += 1

                prog.progress((k + 1) / len(symbols))
                status_box.write(
                    f"Scanning NSE EQ: {k+1:,}/{len(symbols):,} "
                    f"• confirmed {len(confirmed_rows)} "
                    f"• in progress {len(progress_rows)}"
                )
                time.sleep(0.08)

            prog.empty()
            status_box.empty()

            # Keep real data only. No fallback/sample stocks.
            confirmed_df = pd.DataFrame(confirmed_rows)
            progress_df = pd.DataFrame(progress_rows)

            st.session_state["confirmed_df"] = confirmed_df
            st.session_state["progress_df"] = progress_df
            st.session_state["scan_failures"] = failures
            st.session_state["scanned"] = True

    if st.session_state.get("scanned", False):
        confirmed_df = st.session_state.get("confirmed_df", pd.DataFrame())
        progress_df = st.session_state.get("progress_df", pd.DataFrame())

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 Confirmed", len(confirmed_df))
        m2.metric("🟡 In Progress", len(progress_df))
        m3.metric("Data requests failed", st.session_state.get("scan_failures", 0))

        if not confirmed_df.empty:
            st.subheader("🟢 CONFIRMED BREAKOUT")
            display_cols = [
                "Symbol", "Company", "Close", "ATH", "Fall from ATH %",
                "EMA 144", "Fib 0.50", "Fib gap %", "Trendline",
                "Breakout month", "Weekly swing highs", "Status"
            ]
            display_cols = [c for c in display_cols if c in confirmed_df.columns]
            confirmed_df = confirmed_df.sort_values(
                ["Fall from ATH %", "Fib gap %"], ascending=[False, False]
            )
            st.dataframe(
                confirmed_df[display_cols],
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "📥 Download confirmed CSV",
                confirmed_df.to_csv(index=False),
                "nse_confirmed_monthly_breakouts.csv",
                "text/csv"
            )
        else:
            st.info("No confirmed breakout currently satisfies all filters.")

        st.divider()

        if not progress_df.empty:
            st.subheader("🟡 CURRENT-MONTH BREAKOUT IN PROGRESS")
            display_cols = [
                "Symbol", "Company", "Close", "ATH", "Fall from ATH %",
                "EMA 144", "Fib 0.50", "Fib gap %", "Trendline",
                "Breakout month", "Weekly swing highs", "Status"
            ]
            display_cols = [c for c in display_cols if c in progress_df.columns]
            progress_df = progress_df.sort_values(
                ["Fall from ATH %", "Fib gap %"], ascending=[False, False]
            )
            st.dataframe(
                progress_df[display_cols],
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "📥 Download in-progress CSV",
                progress_df.to_csv(index=False),
                "nse_in_progress_monthly_breakouts.csv",
                "text/csv"
            )
        else:
            st.info("No current-month breakout-in-progress setup currently satisfies all filters.")

        st.divider()
        st.subheader("🔎 Inspect a stock")
        all_symbols = sorted(set(
            confirmed_df.get("Symbol", pd.Series(dtype=str)).astype(str).tolist()
            + progress_df.get("Symbol", pd.Series(dtype=str)).astype(str).tolist()
        ))

        if all_symbols:
            selected = st.selectbox("Select a qualifying stock", all_symbols)
            try:
                raw = nse_daily_history(
                    selected,
                    (date.today() - timedelta(days=365 * 25)).strftime("%d-%m-%Y"),
                    date.today().strftime("%d-%m-%Y")
                )
                company = ""
                if "NAME OF COMPANY" in master.columns:
                    matches = master.loc[
                        master[symbol_col].astype(str).str.strip().eq(selected),
                        "NAME OF COMPANY"
                    ]
                    if not matches.empty:
                        company = str(matches.iloc[0])

                show_chart(
                    raw, selected, company,
                    fib_tol, break_buffer, ath_fall_min,
                    weekly_left, weekly_right, min_weekly_gap
                )
            except Exception as e:
                st.warning(f"Chart data could not be loaded: {e}")

        st.caption(
            "The scanner uses only data returned by NSE. "
            "If NSE blocks/rate-limits the deployment, the app does not substitute sample stocks."
        )
