import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Advanced Stock Analyzer", layout="wide")
st.title("📊 Multi-Indicator Accessible Stock Analyzer")
st.markdown("High-contrast, texture-coded dashboard designed for red-green color blindness.")

# ============================================================
# INTERVAL CONFIGURATION
# Each entry defines how we talk to yfinance and whether there's
# enough data at that resolution to compute 50/200-day averages.
# ============================================================
PERIOD_CONFIG = {
    "1 Day":    {"yf_period": "1d",  "yf_interval": "5m",  "show_ma": False, "show_bb": False},
    "5 Days":   {"yf_period": "5d",  "yf_interval": "15m", "show_ma": False, "show_bb": False},
    "1 Month":  {"yf_period": "1mo", "yf_interval": "1d",  "show_ma": False, "show_bb": True},
    "1 Year":   {"yf_period": "1y",  "yf_interval": "1d",  "show_ma": True,  "show_bb": True},
    "2 Years":  {"yf_period": "2y",  "yf_interval": "1d",  "show_ma": True,  "show_bb": True},
    "5 Years":  {"yf_period": "5y",  "yf_interval": "1d",  "show_ma": True,  "show_bb": True},
    "10 Years": {"yf_period": "10y", "yf_interval": "1d",  "show_ma": True,  "show_bb": True},
}

# For anything that needs moving averages, pull extra lookback so the
# 200-day MA has real data behind it on day 1 of the display window,
# then trim back down to the period the user actually asked for.
MA_BUFFER_DAYS = 320

# Colorblind-safe palette (Okabe-Ito), plus dash styles so lines are
# distinguishable even in grayscale, not just by color.
COLOR_PRICE = "#0072B2"      # blue
COLOR_MA50 = "#E69F00"       # orange
COLOR_MA200 = "#000000"      # black
COLOR_BAND = "rgba(0, 114, 178, 0.15)"
COLOR_RSI = "#CC79A7"        # pink
COLOR_VOL = "#BEBEBE"        # neutral grey
COLOR_VOLMA = "#E69F00"      # orange
COLOR_UP = "#0072B2"         # blue = up (never red/green)
COLOR_DOWN = "#E69F00"       # orange = down

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, AAPL, SPY):", value="TSLA").strip().upper()
period_label = st.sidebar.selectbox(
    "Select Time Interval:",
    list(PERIOD_CONFIG.keys()),
    index=3,  # defaults to "1 Year"
)

if st.sidebar.button("🔄 Refresh Live Data"):
    st.cache_data.clear()
    st.toast("Fetching latest live market data...")

config = PERIOD_CONFIG[period_label]


# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=60)
def fetch_stock_data(symbol: str, yf_period: str, yf_interval: str, need_ma_buffer: bool):
    """
    Fetches price history + fundamentals.
    If need_ma_buffer is True, fetches extra daily history so 50/200-day
    averages are valid from the very first displayed candle.
    """
    if need_ma_buffer:
        # Convert period string (e.g. "2y", "10y") into a start date and
        # pad it with a buffer for MA200 warm-up.
        num = int("".join(ch for ch in yf_period if ch.isdigit()))
        unit = "".join(ch for ch in yf_period if ch.isalpha())
        days = num * 365 if unit == "y" else num * 30
        start_date = datetime.today() - timedelta(days=days + MA_BUFFER_DAYS)
        df = yf.download(symbol, start=start_date, interval=yf_interval, auto_adjust=True, progress=False)
    else:
        df = yf.download(symbol, period=yf_period, interval=yf_interval, auto_adjust=True, progress=False)

    # Flatten yfinance's MultiIndex columns (happens for both single & multi-ticker calls
    # depending on yfinance version).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize the index to timezone-naive. This is the root cause of the
    # crash on interval switches: intraday data comes back tz-aware
    # (exchange timezone), daily data sometimes doesn't, and slicing a
    # tz-aware index with a tz-naive datetime raises a TypeError.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Fundamentals (defensive - never let this crash the app)
    fundamentals = {"Trailing P/E": "N/A", "Forward P/E": "N/A", "Name": symbol}
    try:
        info = yf.Ticker(symbol).info
        if isinstance(info, dict):
            fundamentals["Trailing P/E"] = info.get("trailingPE", "N/A")
            fundamentals["Forward P/E"] = info.get("forwardPE", "N/A")
            fundamentals["Name"] = info.get("shortName", symbol)
    except Exception:
        pass

    return df, fundamentals


def clean_pe(val):
    if val is None or val == "N/A":
        return "N/A"
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "N/A"


# ============================================================
# MAIN APP
# ============================================================
if not ticker:
    st.info("Enter a ticker symbol in the sidebar to begin.")
    st.stop()

try:
    raw_data, fundamentals = fetch_stock_data(
        ticker, config["yf_period"], config["yf_interval"], config["show_ma"]
    )
except Exception as e:
    st.error(f"Couldn't fetch data for '{ticker}': {e}")
    st.stop()

if raw_data.empty:
    st.error(f"No data found for '{ticker}'. Please verify the ticker symbol.")
    st.stop()

data = raw_data.copy()

# Guard against unexpected missing columns before doing any math
required_cols = {"Open", "High", "Low", "Close", "Volume"}
if not required_cols.issubset(set(data.columns)):
    st.error("Unexpected data format returned by Yahoo Finance. Try refreshing or a different ticker.")
    st.stop()

# --- INDICATORS (computed on the full/buffered dataset first) ---
has_ma = config["show_ma"] and len(data) >= 200
has_short_ma = config["show_ma"] and len(data) >= 50 and not has_ma

if config["show_ma"]:
    data["MA50"] = data["Close"].rolling(window=50, min_periods=50).mean()
    data["MA200"] = data["Close"].rolling(window=200, min_periods=200).mean()

if config["show_bb"]:
    bb_mid = data["Close"].rolling(window=20, min_periods=20).mean()
    bb_std = data["Close"].rolling(window=20, min_periods=20).std()
    data["BB_Mid"] = bb_mid
    data["BB_Upper"] = bb_mid + (bb_std * 2)
    data["BB_Lower"] = bb_mid - (bb_std * 2)

# RSI (works fine at any interval/resolution)
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
rs = gain / loss.replace(0, np.nan)
data["RSI"] = 100 - (100 / (1 + rs))

data["VolMA20"] = data["Volume"].rolling(window=20, min_periods=1).mean()

# --- TRIM BACK DOWN to the period the user actually selected ---
# (Only relevant when we fetched extra buffer history for MA warm-up.)
if config["show_ma"]:
    num = int("".join(ch for ch in config["yf_period"] if ch.isdigit()))
    unit = "".join(ch for ch in config["yf_period"] if ch.isalpha())
    display_days = num * 365 if unit == "y" else num * 30
    display_start = datetime.today() - timedelta(days=display_days)
    data = data.loc[data.index >= display_start]

if data.empty:
    st.error("Not enough historical data to display this interval for this ticker.")
    st.stop()

# ============================================================
# HEADER METRICS (no red/green — arrows + labeled color instead)
# ============================================================
latest_close = float(data["Close"].iloc[-1])
prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else latest_close
change = latest_close - prev_close
pct_change = (change / prev_close * 100) if prev_close else 0
direction_symbol = "▲" if change >= 0 else "▼"
direction_word = "Up" if change >= 0 else "Down"
direction_color = COLOR_UP if change >= 0 else COLOR_DOWN

col1, col2, col3 = st.columns(3)
col1.metric("Last Price", f"${latest_close:,.2f}")
col2.markdown(
    f"<div style='font-size:14px;color:gray;'>Change</div>"
    f"<div style='font-size:28px;font-weight:600;color:{direction_color};'>"
    f"{direction_symbol} {direction_word} ${abs(change):,.2f}</div>",
    unsafe_allow_html=True,
)
col3.markdown(
    f"<div style='font-size:14px;color:gray;'>% Change</div>"
    f"<div style='font-size:28px;font-weight:600;color:{direction_color};'>"
    f"{direction_symbol} {abs(pct_change):.2f}%</div>",
    unsafe_allow_html=True,
)

if not config["show_ma"]:
    st.caption(
        f"ℹ️ 50-day / 200-day moving averages need at least ~200 daily bars of history, "
        f"so they're hidden for the **{period_label}** view. Switch to 1 Year or longer to see them."
    )
elif not has_ma:
    st.caption("ℹ️ Not quite enough history yet for a full 200-day average — showing available data only.")

# ============================================================
# CHART: 3-PANEL DASHBOARD
# ============================================================
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.55, 0.2, 0.25],
)

# --- PANEL 1: Price, Bollinger Bands, Moving Averages ---
if config["show_bb"] and "BB_Upper" in data.columns:
    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Upper"], mode="lines",
        name="Upper Bollinger Band", line=dict(color=COLOR_BAND, width=1),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Lower"], mode="lines",
        name="Bollinger Bands (20d)", fill="tonexty",
        fillcolor=COLOR_BAND, line=dict(color=COLOR_BAND, width=1),
    ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=data.index, y=data["Close"], mode="lines",
    name="Close Price", line=dict(color=COLOR_PRICE, width=2.5)
), row=1, col=1)

if config["show_ma"]:
    fig.add_trace(go.Scatter(
        x=data.index, y=data["MA50"], mode="lines",
        name="50-Day MA", line=dict(color=COLOR_MA50, width=2, dash="dash"),
        connectgaps=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["MA200"], mode="lines",
        name="200-Day MA", line=dict(color=COLOR_MA200, width=2, dash="dot"),
        connectgaps=False,
    ), row=1, col=1)

# --- PANEL 2: RSI ---
fig.add_trace(go.Scatter(
    x=data.index, y=data["RSI"], mode="lines",
    name="RSI (14)", line=dict(color=COLOR_RSI, width=2)
), row=2, col=1)

fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
              line=dict(color="#000000", width=1.5, dash="dot"), row=2, col=1)
fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
              line=dict(color="#000000", width=1.5, dash="dot"), row=2, col=1)

# --- PANEL 3: Volume ---
fig.add_trace(go.Bar(
    x=data.index, y=data["Volume"],
    name="Volume", marker_color=COLOR_VOL, showlegend=True
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=data.index, y=data["VolMA20"], mode="lines",
    name="20-Bar Volume Avg", line=dict(color=COLOR_VOLMA, width=1.5)
), row=3, col=1)

fig.update_layout(
    title=f"{ticker} — {period_label} Market Analysis",
    hovermode="x unified", template="plotly_white",
    height=750,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=90),
)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[10, 90], row=2, col=1)
fig.update_yaxes(title_text="Volume", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# VALUATION TABLE
# ============================================================
st.subheader("🔑 Company Valuation Summary")

valuation_df = pd.DataFrame({
    "Valuation Metric": ["Company", "Historical (Trailing) P/E", "Forward (Expected) P/E"],
    "Current Value": [
        fundamentals.get("Name", ticker),
        clean_pe(fundamentals.get("Trailing P/E")),
        clean_pe(fundamentals.get("Forward P/E")),
    ],
    "What it Means": [
        "The company name on file with Yahoo Finance.",
        "Based on actual net earnings over the past 12 months.",
        "Based on Wall Street analyst consensus forecasts for the next 12 months.",
    ],
})

st.dataframe(valuation_df, use_container_width=True, hide_index=True)
