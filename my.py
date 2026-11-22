import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import time
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# MY STRATEGY - DAILY SWING SCANNER (PAPER TRADING)
# Cloud-ready reconstruction. No broker order placement.
# =========================================================

st.set_page_config(
    page_title="MY STRATEGY",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "paper_trades.db"
NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

CAPITAL = 500000
RISK_PER_TRADE = 0.01
MAX_HOLDING_DAYS = 10
ADX_MIN = 20
BATCH_SIZE = 25
PERIOD = "6mo"
INTERVAL = "1d"

st.markdown("""
<style>
    .stApp {background-color:#0e1117;color:#f5f5f5;}
    div[data-testid="stMetric"] {
        background:#171b24;border:1px solid #303744;
        padding:12px;border-radius:10px;
    }
    .small-note {color:#aab3c2;font-size:12px;}
    div.stButton > button {border-radius:8px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("📈 MY STRATEGY")
st.caption("Daily stock swing scanner | Paper trading only | No automatic broker orders")

# -------------------- Database --------------------
def db_connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            signal_price REAL,
            buy_above REAL,
            stop_loss REAL,
            target1 REAL,
            target2 REAL,
            status TEXT DEFAULT 'WAITING',
            entry_date TEXT,
            entry_price REAL,
            exit_date TEXT,
            exit_price REAL,
            result TEXT,
            updated_at TEXT
        )
    """)
    con.commit()
    return con

def load_trades():
    con = db_connect()
    df = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY id DESC", con)
    con.close()
    return df

def save_signal(row):
    con = db_connect()
    exists = con.execute(
        "SELECT id FROM paper_trades WHERE symbol=? AND signal_date=?",
        (row["Symbol"], str(row["Signal Date"]))
    ).fetchone()
    if not exists:
        con.execute("""
            INSERT INTO paper_trades
            (symbol, signal_date, signal_price, buy_above, stop_loss,
             target1, target2, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'WAITING', ?)
        """, (
            row["Symbol"], str(row["Signal Date"]), float(row["Price"]),
            float(row["Buy Above"]), float(row["SL"]), float(row["T1"]),
            float(row["T2"]), datetime.now().isoformat(timespec="seconds")
        ))
        con.commit()
    con.close()

# -------------------- Indicators --------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_indicators(df):
    d = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    close = d["Close"]
    high = d["High"]
    low = d["Low"]

    d["EMA20"] = ema(close, 20)
    d["EMA50"] = ema(close, 50)

    fast = ema(close, 12)
    slow = ema(close, 26)
    d["MACD"] = fast - slow
    d["MACD_SIGNAL"] = ema(d["MACD"], 9)
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIGNAL"]
    d["MACD_CROSS_UP"] = (
        (d["MACD_HIST"] > 0) &
        (d["MACD_HIST"].shift(1) <= 0)
    )

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=d.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=d.index
    )
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["PLUS_DI"] = plus_di
    d["MINUS_DI"] = minus_di
    d["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Supertrend (7, 2)
    st_period, multiplier = 7, 2.0
    atr_st = tr.ewm(alpha=1/st_period, adjust=False).mean()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr_st
    lower = hl2 - multiplier * atr_st
    final_upper = upper.copy()
    final_lower = lower.copy()
    for i in range(1, len(d)):
        if pd.isna(final_upper.iloc[i-1]) or pd.isna(final_lower.iloc[i-1]):
            continue
        if upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        if lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    st_line = pd.Series(index=d.index, dtype=float)
    st_dir = pd.Series(index=d.index, dtype=int)
    for i in range(len(d)):
        if i == 0:
            st_dir.iloc[i] = 1
            st_line.iloc[i] = final_lower.iloc[i]
            continue
        prev_dir = st_dir.iloc[i-1]
        if prev_dir == 1:
            if close.iloc[i] < final_lower.iloc[i]:
                st_dir.iloc[i] = -1
                st_line.iloc[i] = final_upper.iloc[i]
            else:
                st_dir.iloc[i] = 1
                st_line.iloc[i] = final_lower.iloc[i]
        else:
            if close.iloc[i] > final_upper.iloc[i]:
                st_dir.iloc[i] = 1
                st_line.iloc[i] = final_lower.iloc[i]
            else:
                st_dir.iloc[i] = -1
                st_line.iloc[i] = final_upper.iloc[i]
    d["SUPERTREND"] = st_line
    d["ST_DIR"] = st_dir
    d["VOL_AVG20"] = d["Volume"].rolling(20).mean()
    return d

# -------------------- Universe and data --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty500():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(NIFTY500_URL, headers=headers, timeout=20)
    response.raise_for_status()
    raw = pd.read_csv(pd.io.common.StringIO(response.text))
    symbol_col = "Symbol" if "Symbol" in raw.columns else raw.columns[0]
    names = raw[symbol_col].dropna().astype(str).str.strip().tolist()
    return [s + ".NS" for s in names if s]

def normalize_download(data, symbol):
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if symbol in data.columns.get_level_values(0):
                data = data[symbol]
            elif symbol in data.columns.get_level_values(-1):
                data = data.xs(symbol, axis=1, level=-1)
            else:
                data.columns = data.columns.get_level_values(0)
        except Exception:
            return pd.DataFrame()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in data.columns for c in needed):
        return pd.DataFrame()
    return data[needed].dropna(subset=["Open", "High", "Low", "Close"])

@st.cache_data(ttl=900, show_spinner=False)
def download_batch(symbols):
    try:
        raw = yf.download(
            tickers=symbols,
            period=PERIOD,
            interval=INTERVAL,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
        return raw
    except Exception:
        return pd.DataFrame()

def completed_daily_data(df):
    if df.empty:
        return df
    # During market hours, ignore today's still-forming daily candle.
    today = datetime.now().date()
    try:
        last_date = pd.Timestamp(df.index[-1]).date()
        if last_date >= today:
            df = df.iloc[:-1]
    except Exception:
        pass
    return df

def scan_symbol(symbol, batch_data):
    data = normalize_download(batch_data, symbol)
    if data.empty or len(data) < 60:
        return None
    data = completed_daily_data(data)
    if len(data) < 60:
        return None
    d = calculate_indicators(data).dropna(subset=["ADX", "ATR", "EMA50", "MACD_HIST"])
    if len(d) < 5:
        return None

    latest = d.iloc[-1]
    recent = d.tail(2)
    cross_recent = bool(recent["MACD_CROSS_UP"].any())
    bullish = (
        cross_recent and
        latest["ST_DIR"] == 1 and
        latest["ADX"] >= ADX_MIN and
        latest["PLUS_DI"] > latest["MINUS_DI"]
    )
    if not bullish or latest["Close"] < 250 or latest["Close"] > 1000:
        return None

    atr = float(latest["ATR"])
    buy_above = float(latest["High"]) * 1.001
    sl = buy_above - 1.5 * atr
    t1 = buy_above + 2.0 * atr
    t2 = buy_above + 3.0 * atr

    score = 0
    score += int(latest["ADX"] >= 25)
    score += int(latest["PLUS_DI"] > latest["MINUS_DI"])
    score += int(latest["EMA20"] > latest["EMA50"])
    score += int(latest["Close"] > latest["EMA20"])
    score += int(latest["Volume"] > latest["VOL_AVG20"])
    score += int(latest["ST_DIR"] == 1)

    return {
        "Symbol": symbol.replace(".NS", ""),
        "Signal Date": pd.Timestamp(d.index[-1]).date().isoformat(),
        "Price": round(float(latest["Close"]), 2),
        "Buy Above": round(buy_above, 2),
        "SL": round(sl, 2),
        "T1": round(t1, 2),
        "T2": round(t2, 2),
        "ADX": round(float(latest["ADX"]), 2),
        "+DI": round(float(latest["PLUS_DI"]), 2),
        "-DI": round(float(latest["MINUS_DI"]), 2),
        "Score": score,
        "Strength": "STRONG" if score >= 5 else "NORMAL",
    }

# -------------------- UI --------------------
top1, top2, top3 = st.columns(3)
top1.metric("Paper Capital", "₹5,00,000")
top2.metric("Risk / Trade", "1%")
top3.metric("Holding Period", "10 days")

st.info(
    "Signals use completed daily candles. Buy Above is a trigger level, "
    "not an instruction to buy immediately. Confirm price action before any real trade."
)

col_a, col_b = st.columns([1, 1])
with col_a:
    run_scan = st.button("🔄 Refresh Scanner", use_container_width=True, type="primary")
with col_b:
    show_all = st.button("📋 Show Paper Log", use_container_width=True)

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if run_scan:
    try:
        with st.spinner("Downloading Nifty 500 data and scanning..."):
            symbols = get_nifty500()
            found = []
            progress = st.progress(0)
            status = st.empty()
            for start in range(0, len(symbols), BATCH_SIZE):
                batch_symbols = symbols[start:start+BATCH_SIZE]
                batch_data = download_batch(batch_symbols)
                for symbol in batch_symbols:
                    result = scan_symbol(symbol, batch_data)
                    if result:
                        found.append(result)
                        save_signal(result)
                progress.progress(min((start + len(batch_symbols)) / len(symbols), 1.0))
                status.caption(f"Scanned {min(start + len(batch_symbols), len(symbols))} / {len(symbols)} stocks")
            progress.empty()
            status.empty()
            result_df = pd.DataFrame(found)
            if not result_df.empty:
                result_df = result_df.sort_values(["Score", "ADX"], ascending=False)
            st.session_state.scan_results = result_df
            st.session_state.last_scan = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    except Exception as e:
        st.error(f"Scanner error: {e}")

if st.session_state.last_scan:
    st.caption(f"Last scan: {st.session_state.last_scan}")

if st.session_state.scan_results is not None:
    results = st.session_state.scan_results
    st.subheader(f"Today's Shortlist ({len(results)})")
    if results.empty:
        st.warning("No stocks matched all conditions in this scan. Try Refresh Scanner later.")
    else:
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download shortlist CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="my_strategy_shortlist.csv",
            mime="text/csv"
        )

st.divider()
st.subheader("📒 Paper Trading Log")
trades = load_trades()
if trades.empty:
    st.caption("No paper signals saved yet. Run Refresh Scanner to create the first log.")
else:
    st.dataframe(trades, use_container_width=True, hide_index=True)
    st.download_button(
        "Download paper log CSV",
        data=trades.to_csv(index=False).encode("utf-8"),
        file_name="my_strategy_paper_log.csv",
        mime="text/csv"
    )

with st.expander("Strategy rules and important limitations"):
    st.markdown("""
    - Universe: Nifty 500 constituents.
    - Price filter: ₹250–₹1,000.
    - Signal: MACD bullish crossover within the latest two completed daily candles,
      bullish Supertrend (7,2), ADX at least 20, and +DI above -DI.
    - Strength score: ADX, DI direction, EMA20/EMA50, close/EMA20,
      volume vs 20-day average, and Supertrend.
    - Buy Above = latest completed daily High × 1.001.
    - SL = Buy Above − 1.5 × ATR(14); T1 = Buy Above + 2 × ATR;
      T2 = Buy Above + 3 × ATR.
    - This version records signals in a local SQLite database. On free cloud hosting,
      local files/database may be reset or lost when the app restarts or redeploys.
      Use a persistent database service before relying on long-term paper records.
    - This is a research/paper-trading tool, not financial advice. It does not place orders.
    """)

st.caption("MY STRATEGY | Paper trading only | Market data supplied by Yahoo Finance")
