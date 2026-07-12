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
    # Add extra padding days to accurately calculate 200-day averages
    start_date = end_date - timedelta(days=days_back + 300) 
    df = yf.download(symbol, start=start_date, end=end_date)
    
    # Fetch fundamental data (P/E ratios)
    ticker_obj = yf.Ticker(symbol)
    info = ticker_obj.info
    fundamentals = {
        "Trailing P/E": info.get("trailingPE", "N/A"),
        "Forward P/E": info.get("forwardPE", "N/A"),
        "5-Year Avg P/E": info.get("trailingPegRatio", "N/A") # Using PEG as an alternative anchor if historical average isn't available
    }
    
    return df, fundamentals

if ticker:
    try:
        raw_data, fundamentals = fetch_stock_data(ticker, years * 365)
        
        if not raw_data.empty:
            data = raw_data.copy()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # --- TECHNICAL INDICATOR CALCULATIONS ---
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()
            
            data['BB_Mid'] = data['Close'].rolling(window=20).mean()
            data['BB_Std'] = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Mid'] + (data['BB_Std'] * 2)
            data['BB_Lower'] = data['BB_Mid'] - (data['BB_Std'] * 2)
            
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            data['VolMA20'] = data['Volume'].rolling(window=20).mean()
            
            display_start = datetime.today() - timedelta(days=years * 365)
            data = data.loc[display_start:]
            
            # --- CONSTRUCTING THE 3-PANEL DASHBOARD ---
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_width=[0.2, 0.2, 0.6] 
            )
            
            # PANEL 1: PRICE, AVERAGES & BOLLINGER BANDS
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Upper'], mode='lines',
                name='Upper Bollinger Band', line=dict(color='rgba(0, 114, 178, 0.2)', width=1),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Lower'], mode='lines',
                name='Bollinger Bands', fill='tonexty', 
                fillcolor='rgba(0, 114, 178, 0.05)', 
                line=dict(color='rgba(0, 114, 178, 0.2)', width=1),
                legendgroup='bands'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'], mode='lines', 
                name='Close Price', line=dict(color='#0072B2', width=2.5, dash='solid')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA50'], mode='lines', 
                name='50-Day MA', line=dict(color='#E69F00', width=2, dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA200'], mode='lines', 
                name='200-Day MA', line=dict(color='#000000', width=2, dash='dot')
            ), row=1, col=1)
            
            # PANEL 2: RELATIVE STRENGTH INDEX (RSI)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['RSI'], mode='lines', 
                name='RSI (14)', line=dict(color='#CC79A7', width=2)
            ), row=2, col=1)
            
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
                          line=dict(color="#000000", width=1.5, dash="dot"), row=2, col=1)
            fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
                          line=dict(color="#000000", width=1.5, dash="dot"), row=2, col=1)
            
            # PANEL 3: VOLUME
            fig.add_trace(go.Bar(
                x=data.index, y=data['Volume'], 
                name='Daily Volume', marker_color='#E5E5E5',
                showlegend=True
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['VolMA20'], mode='lines', 
                name='20-Day Volume Avg', line=dict(color='#E69F00', width=1.5)
            ), row=3, col=1)
            
            fig.update_layout(
                title=f"{ticker} Comprehensive Market Analysis",
                hovermode="x unified", template="plotly_white",
                height=750, 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                yaxis_title="Price ($)", yaxis2_title="RSI", yaxis3_title="Volume",
                yaxis2=dict(range=[10, 90])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- NEW SECTION: VALUATION DATA TABLE ---
            st.subheader("🔑 Company Valuation Summary")
            
            # Formatting values for clean table display
            pe_trailing = f"{fundamentals['Trailing P/E']:.2f}" if isinstance(fundamentals['Trailing P/E'], (int, float)) else "N/A"
            pe_forward = f"{fundamentals['Forward P/E']:.2f}" if isinstance(fundamentals['Forward P/E'], (int, float)) else "N/A"
            
            # Build a stark, accessible text dataframe
            valuation_df = pd.DataFrame({
                "Valuation Metric": ["Historical (Trailing) P/E", "Forward (Expected) P/E"],
                "Current Value": [pe_trailing, pe_forward],
                "What it Means": [
                    f"Based on actual net earnings over the past 12 months.",
                    f"Based on Wall Street analyst consensus forecasts for the next 12 months."
                ]
            })
            
            # Display as a high-contrast text table
            st.dataframe(valuation_df, use_container_width=True, hide_index=True)
            
        else:
            st.error("No data found. Please verify the ticker symbol.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
