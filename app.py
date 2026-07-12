import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page setup for optimal screen sizing on mobile/Tesla
st.set_page_config(page_title="Advanced Stock Analyzer", layout="wide")
st.title("📊 Multi-Indicator Accessible Stock Analyzer")
st.markdown("Optimized high-contrast, textured dashboard for red-green color blindness.")

# Sidebar user inputs
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, AAPL, SPY):", value="TSLA").upper()
years = st.sidebar.slider("Select Data Range (Years):", min_value=1, max_value=5, value=2)

# Clear data cache instantly when user requests fresh market data
if st.sidebar.button("🔄 Refresh Live Data"):
    st.cache_data.clear()
    st.toast("Fetching latest live market data...")

# Automated background cache reset after 60 seconds
@st.cache_data(ttl=60)
def fetch_stock_data(symbol, days_back):
    end_date = datetime.today()
    # Add extra padding days to accurately calculate 200-day averages at the start of our timeline
    start_date = end_date - timedelta(days=days_back + 300) 
    df = yf.download(symbol, start=start_date, end=end_date)
    return df, start_date

if ticker:
    try:
        raw_data, calculated_start = fetch_stock_data(ticker, years * 365)
        
        if not raw_data.empty:
            data = raw_data.copy()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # --- TECHNICAL INDICATOR CALCULATIONS ---
            # 1. Moving Averages
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()
            
            # 2. Bollinger Bands (20-day SMA +/- 2 Standard Deviations)
            data['BB_Mid'] = data['Close'].rolling(window=20).mean()
            data['BB_Std'] = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Mid'] + (data['BB_Std'] * 2)
            data['BB_Lower'] = data['BB_Mid'] - (data['BB_Std'] * 2)
            
            # 3. Relative Strength Index (RSI - 14 day)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # 4. Volume Moving Average
            data['VolMA20'] = data['Volume'].rolling(window=20).mean()
            
            # Filter the dataset down to the specific window requested by the slider
            display_start = datetime.today() - timedelta(days=years * 365)
            data = data.loc[display_start:]
            
            # --- CONSTRUCTING THE 3-PANEL DASHBOARD ---
            # Row 1 (Price & Bands): 60% height | Row 2 (RSI): 20% | Row 3 (Volume): 20%
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_width=[0.2, 0.2, 0.6] 
            )
            
            # PANEL 1: PRICE, AVERAGES & BOLLINGER BANDS
            # Bollinger Upper Band (Faint Blue Line)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Upper'], mode='lines',
                name='Upper Bollinger Band', line=dict(color='rgba(0, 114, 178, 0.2)', width=1),
                showlegend=False
            ), row=1, col=1)
            
            # Bollinger Lower Band (Faint Blue Line filled down to Upper to create a channel)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Lower'], mode='lines',
                name='Bollinger Bands', fill='tonexty', 
                fillcolor='rgba(0, 114, 178, 0.05)', # Ultra translucent light blue shading
                line=dict(color='rgba(0, 114, 178, 0.2)', width=1),
                legendgroup='bands'
            ), row=1, col=1)
            
            # Stock Price: Solid Deep Blue
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'], mode='lines', 
                name='Close Price', line=dict(color='#0072B2', width=2.5, dash='solid')
            ), row=1, col=1)
            
            # 50-Day MA: Dashed Orange
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA50'], mode='lines', 
                name='50-Day MA', line=dict(color='#E69F00', width=2, dash='dash')
            ), row=1, col=1)
            
            # 200-Day MA: Dotted Black
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA200'], mode='lines', 
                name='200-Day MA', line=dict(color='#000000', width=2, dash='dot')
            ), row=1, col=1)
            
            # PANEL 2: RELATIVE STRENGTH INDEX (RSI)
            # RSI Line: Thick Purple
            fig.add_trace(go.Scatter(
                x=data.index, y=data['RSI'], mode='lines', 
                name='RSI (14)', line=dict(color='#CC79A7', width=2)
            ), row=2, col=1)
            
            # Overbought (70) Marker: Horizontal Dotted Line
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
                          line=dict(color="#000000", width=1.5, dash="dot"), row=2, col=1)
            # Oversold (30) Marker: Horizontal Dotted Line
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
                          line=dict(color="#000000", width=1.5, dash="dot"), row=2, col=1)
            
            # PANEL 3: VOLUME
            # Volume Bars: Uniform Light Gray
            fig.add_trace(go.Bar(
                x=data.index, y=data['Volume'], 
                name='Daily Volume', marker_color='#E5E5E5',
                showlegend=True
            ), row=3, col=1)
            
            # 20-day Volume Average: Solid Orange
            fig.add_trace(go.Scatter(
                x=data.index, y=data['VolMA20'], mode='lines', 
                name='20-Day Volume Avg', line=dict(color='#E69F00', width=1.5)
            ), row=3, col=1)
            
            # GLOBAL LAYOUT SETTINGS
            fig.update_layout(
                title=f"{ticker} Comprehensive Market Analysis",
                hovermode="x unified", template="plotly_white",
                height=750, # Expanded height to neatly fit all three layout sections
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                yaxis_title="Price ($)", yaxis2_title="RSI", yaxis3_title="Volume",
                yaxis2=dict(range=[10, 90]) # Clips RSI boundaries cleanly
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # DATA SUMMARY TEXT CARDS
            latest_close = float(data['Close'].iloc[-1])
            latest_ma50 = float(data['MA50'].iloc[-1]) if not pd.isna(data['MA50'].iloc[-1]) else None
            latest_rsi = float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else None
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Close", f"${latest_close:.2f}")
            col2.metric("50-Day MA", f"${latest_ma50:.2f}" if latest_ma50 else "Calculating...")
            col3.metric("RSI (14)", f"{latest_rsi:.1f}" if latest_rsi else "Calculating...")
            
        else:
            st.error("No data found. Please verify the ticker symbol.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
