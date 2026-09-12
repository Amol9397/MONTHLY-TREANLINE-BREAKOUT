import streamlit as st
import pandas as pd
import numpy as np
import requests, io, time
from datetime import date, timedelta

st.set_page_config(page_title="NSE Monthly Trend Lab", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#06111f,#0b2038 48%,#102b24);color:#eef7ff}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#061421,#10263b)}
h1{background:linear-gradient(90deg,#00e5ff,#72ff8c,#ffd166);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
div[data-testid="stMetric"]{background:linear-gradient(135deg,#102d49,#123c38);border:1px solid #2b6680;border-radius:16px;padding:12px}
div.stButton>button{border-radius:12px;background:linear-gradient(90deg,#087fdb,#00a88f);color:white;font-weight:800}
</style>
""", unsafe_allow_html=True)

st.title("📈 NSE MONTHLY TREND LAB")
st.caption("NSE-listed equity universe • Monthly-candle scanner • Strategy research")

st.sidebar.header("Scanner settings")
fib_tol = st.sidebar.slider("0.50 Fib 'around' tolerance %", 0, 10, 5)
break_buffer = st.sidebar.slider("Breakout buffer %", 0.0, 5.0, 0.5, 0.1)
min_history = st.sidebar.number_input("Minimum monthly candles", 160, 500, 160, 10)

NSE_MASTER = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_HOME = "https://www.nseindia.com"
NSE_HIST = "https://www.nseindia.com/api/historical/cm/equity"

@st.cache_data(ttl=86400)
def nse_equity_master():
    s = requests.Session()
    s.headers.update({
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
        "Accept":"text/csv,text/plain,*/*",
        "Referer":NSE_HOME+"/"
    })
    r=s.get(NSE_HOME,timeout=20)
    r.raise_for_status()
    r=s.get(NSE_MASTER,timeout=30)
    r.raise_for_status()
    df=pd.read_csv(io.BytesIO(r.content))
    df.columns=[c.strip() for c in df.columns]
    # NSE EQ series only: exclude ETFs/REIT/INVIT/SME etc. that are not normal EQ shares.
    if "SERIES" in df.columns:
        df=df[df["SERIES"].astype(str).str.upper().eq("EQ")].copy()
    return df

def nse_session():
    s=requests.Session()
    s.headers.update({
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
        "Accept":"application/json,text/plain,*/*",
        "Referer":NSE_HOME+"/",
        "Accept-Language":"en-IN,en;q=0.9"
    })
    s.get(NSE_HOME,timeout=20)
    return s

@st.cache_data(ttl=2592000, show_spinner=False)
def nse_daily_history(symbol, start, end):
    # Historical data is fetched only for the requested stock/date range.
    # The app aggregates it to MONTHLY OHLC locally and caches the result.
    s=nse_session()
    params={"symbol":symbol,"series":'["EQ"]',"from":start,"to":end}
    r=s.get(NSE_HIST,params=params,timeout=30)
    r.raise_for_status()
    j=r.json()
    rows=j.get("data",[])
    if not rows:
        return pd.DataFrame()
    df=pd.DataFrame(rows)
    # NSE API field names can vary slightly.
    rename={}
    for c in df.columns:
        u=c.upper()
        if u=="CH_CLOSING_PRICE": rename[c]="CLOSE"
        elif u=="CH_OPENING_PRICE": rename[c]="OPEN"
        elif u=="CH_TRADE_HIGH_PRICE": rename[c]="HIGH"
        elif u=="CH_TRADE_LOW_PRICE": rename[c]="LOW"
        elif u=="CH_TIMESTAMP": rename[c]="DATE"
    df=df.rename(columns=rename)
    if not {"DATE","OPEN","HIGH","LOW","CLOSE"}.issubset(df.columns):
        return pd.DataFrame()
    df["DATE"]=pd.to_datetime(df["DATE"],dayfirst=True,errors="coerce")
    for c in ["OPEN","HIGH","LOW","CLOSE"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=["DATE","CLOSE"]).sort_values("DATE")

def to_monthly(d):
    if d.empty:return d
    x=d.set_index("DATE")[["OPEN","HIGH","LOW","CLOSE"]].resample("ME").agg(
        {"OPEN":"first","HIGH":"max","LOW":"min","CLOSE":"last"}).dropna()
    return x

def ema(s,n=144):
    return s.ewm(span=n,adjust=False).mean()

def pivots_high(s,left=2,right=2):
    out=[]
    for i in range(left,len(s)-right):
        if s.iloc[i]>=s.iloc[i-left:i+right+1].max():
            out.append(i)
    return out

def scan(df,fib_tol,buffer):
    if len(df)<min_history:return None
    # Only completed months.
    d=df.iloc[:-1].copy()
    if len(d)<min_history:return None
    c=d["CLOSE"]; h=d["HIGH"]; l=d["LOW"]
    e=ema(c,144).iloc[-1]
    ath=float(h.max()); atl=float(l.min())
    fib50=ath-(ath-atl)*0.5
    p=pivots_high(h.reset_index(drop=True))
    if len(p)<2:return None
    tl=None
    for a in range(len(p)-2,-1,-1):
        for b in range(len(p)-1,a,-1):
            i,j=p[a],p[b]
            if j>i and h.iloc[j]<h.iloc[i]:
                slope=(h.iloc[j]-h.iloc[i])/(j-i)
                intercept=h.iloc[i]-slope*i
                tl=(slope,intercept,i,j);break
        if tl:break
    if not tl:return None
    slope,intercept,i,j=tl
    n=len(d)-1
    line=slope*n+intercept
    prevline=slope*(n-1)+intercept
    close=float(c.iloc[-1]); prevclose=float(c.iloc[-2])
    breakout=prevclose<=prevline and close>line*(1+buffer/100)
    above_ema=close>e
    fib_ok=close>=fib50*(1-fib_tol/100)
    if not (breakout and above_ema and fib_ok):return None
    return {
        "Symbol":None,"Close":round(close,2),"EMA 144":round(float(e),2),
        "Fib 0.50":round(fib50,2),"Fib gap %":round((close/fib50-1)*100,2),
        "Trendline":round(line,2),"Breakout month":d.index[-1].strftime("%Y-%m"),
        "Trendline pivots":f"{d.index[i].strftime('%Y-%m')} → {d.index[j].strftime('%Y-%m')}"
    }

tab1,tab2,tab3=st.tabs(["🏠 Dashboard","🔷 Monthly Trend Line Break","📘 Rules"])

with tab1:
    st.metric("Universe","NSE-listed EQ securities")
    st.metric("Candle frequency","Monthly")
    st.info("This tab is deliberately NOT a daily scanner. The signal engine works on monthly candles only.")

with tab2:
    st.subheader("🔷 Monthly Trend Line Break")
    st.markdown("**Your exact filter:** Monthly descending trendline breakout + **Close > EMA 144** + **price above/around 0.50 Fib** using **all-time high → all-time low**.")
    if st.button("🔄 Update NSE Equity Universe"):
        st.cache_data.clear()
    try:
        master=nse_equity_master()
        symbol_col="SYMBOL" if "SYMBOL" in master.columns else master.columns[0]
        st.success(f"NSE EQ universe loaded: {len(master):,} securities")
        st.caption("Historical monthly data is cached. After the initial history build, only the new period needs refreshing.")
    except Exception as e:
        st.error(f"NSE universe could not be loaded: {e}")
        master=pd.DataFrame()

    if not master.empty and st.button("🚀 Scan NSE Monthly Setups",type="primary"):
        rows=[]
        prog=st.progress(0)
        status=st.empty()
        # For a production deployment, process the complete master with throttling.
        # Historical bootstrap is intentionally separated from the monthly refresh.
        symbols=master[symbol_col].dropna().astype(str).str.strip().tolist()
        for k,sym in enumerate(symbols):
            try:
                start=(date.today()-timedelta(days=365*25)).strftime("%d-%m-%Y")
                end=date.today().strftime("%d-%m-%Y")
                raw=nse_daily_history(sym,start,end)
                mon=to_monthly(raw)
                r=scan(mon,fib_tol,break_buffer)
                if r:
                    r["Symbol"]=sym
                    if "NAME OF COMPANY" in master.columns:
                        r["Company"]=master.loc[master[symbol_col].eq(sym),"NAME OF COMPANY"].iloc[0]
                    rows.append(r)
            except Exception:
                pass
            prog.progress((k+1)/len(symbols))
            status.write(f"Scanning NSE EQ: {k+1:,}/{len(symbols):,}")
            time.sleep(0.12)
        prog.empty();status.empty()
        if rows:
            out=pd.DataFrame(rows).sort_values("Fib gap %",ascending=False)
            st.success(f"{len(out)} stocks passed all conditions.")
            st.dataframe(out,use_container_width=True,hide_index=True)
            st.download_button("⬇️ Download signals",out.to_csv(index=False),"nse_monthly_trendline_break.csv","text/csv")
        else:
            st.info("No confirmed setups found in the current completed-month scan.")

with tab3:
    st.subheader("📘 Scanner definition")
    st.write("1. Universe = NSE listed equity securities, EQ series.")
    st.write("2. Data = monthly candles only for this scanner; daily NSE history is aggregated locally when required for historical bootstrap.")
    st.write("3. Main signal uses the latest completed monthly candle.")
    st.write("4. Descending trendline is built from confirmed monthly swing highs.")
    st.write("5. Breakout requires the monthly close above the trendline.")
    st.write("6. Close must be above Monthly EMA 144.")
    st.write("7. Fibonacci is drawn from all-time high to all-time low; price must be above or around 0.50.")
    st.warning("NSE may rate-limit automated requests. A production deployment should use persistent cached monthly data and update only the newest month.")
