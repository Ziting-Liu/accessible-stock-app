import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page setup for optimal screen sizing on mobile/Tesla
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Accessible Stock Moving Average Analyzer")
st.markdown("Customized high-contrast, textured lines optimized for red-green color blindness.")

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
    start_date = end_date - timedelta(days=days_back)
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

if ticker:
    try:
        data = fetch_stock_data(ticker, years * 365)
        
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # Calculate 50-day and 200-day Simple Moving Averages
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()
            
            # Draw custom accessible plot
            fig = go.Figure()
            
            # Stock Close Price: Solid Line, Deep Blue (#0072B2)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'],
                mode='lines', name='Close Price',
                line=dict(color='#0072B2', width=2, dash='solid')
            ))
            
            # 50-Day Moving Average: Dashed Line, Vibrant Orange (#E69F00)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA50'],
                mode='lines', name='50-Day MA',
                line=dict(color='#E69F00', width=2.5, dash='dash')
            ))
            
            # 200-Day Moving Average: Dotted Line, Matte Black (#000000)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA200'],
                mode='lines', name='200-Day MA',
                line=dict(color='#000000', width=2.5, dash='dot')
            ))
            
            fig.update_layout(
                title=f"{ticker} Live Performance Metrics",
                xaxis_title="Date", yaxis_title="Price ($)",
                hovermode="x unified", template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Summary Cards
            latest_close = float(data['Close'].iloc[-1])
            latest_ma50 = float(data['MA50'].iloc[-1]) if not pd.isna(data['MA50'].iloc[-1]) else None
            latest_ma200 = float(data['MA200'].iloc[-1]) if not pd.isna(data['MA200'].iloc[-1]) else None
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Close", f"${latest_close:.2f}")
            col2.metric("50-Day MA", f"${latest_ma50:.2f}" if latest_ma50 else "Calculating...")
            col3.metric("200-Day MA", f"${latest_ma200:.2f}" if latest_ma200 else "Calculating...")
            
        else:
            st.error("No data found. Please verify the ticker symbol.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
