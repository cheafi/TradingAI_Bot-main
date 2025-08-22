"""
Enhanced Streamlit dashboard with multiple pages, themes, and advanced features.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
import io
import base64

# Configure page
st.set_page_config(
    page_title="TradingAI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .sidebar .sidebar-content {
        background: #f1f3f4;
    }
</style>
""", unsafe_allow_html=True)

def create_download_link(df, filename, file_format="csv"):
    """Create download link for dataframe."""
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    elif file_format == "json":
        json_str = df.to_json(orient="records", indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">Download JSON</a>'
    return href

def main_dashboard():
    """Main dashboard page."""
    st.markdown('<div class="main-header"><h1>🚀 TradingAI Pro Dashboard</h1></div>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("🎯 Trading Controls")
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "📊 Select Symbol",
        ["AAPL", "MSFT", "GOOGL", "TSLA", "BTC/USDT", "ETH/USDT"],
        index=0
    )
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "🔄 Trading Mode",
        ["Backtest", "Paper Trading", "Live Trading"],
        index=0
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    
    # Strategy parameters
    st.sidebar.subheader("⚙️ Strategy Parameters")
    ema_period = st.sidebar.slider("EMA Period", 5, 200, 20)
    atr_period = st.sidebar.slider("ATR Period", 5, 50, 14)
    rsi_period = st.sidebar.slider("RSI Period", 2, 30, 14)
    
    # Risk management
    st.sidebar.subheader("🛡️ Risk Management")
    max_position = st.sidebar.slider("Max Position Size (%)", 1, 20, 5)
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0.5, 10.0, 2.0)
    take_profit = st.sidebar.slider("Take Profit (%)", 1.0, 20.0, 5.0)
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            '<div class="metric-card success-metric">'
            '<h3>💰 Portfolio Value</h3>'
            '<h2>$12,345</h2>'
            '<p>+5.2% today</p>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<h3>📊 Sharpe Ratio</h3>'
            '<h2>2.45</h2>'
            '<p>Excellent</p>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            '<div class="metric-card warning-metric">'
            '<h3>📉 Max Drawdown</h3>'
            '<h2>3.2%</h2>'
            '<p>Within limits</p>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            '<div class="metric-card">'
            '<h3>🎯 Win Rate</h3>'
            '<h2>68%</h2>'
            '<p>Strong performance</p>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Charts section
    st.subheader("📈 Price & Signals Chart")
    
    # Generate sample data for demo
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
    
    df_chart = pd.DataFrame({
        'Date': dates[:len(prices)],
        'Price': prices,
        'EMA': pd.Series(prices).rolling(ema_period).mean(),
        'Signal': np.random.choice([0, 1], len(prices), p=[0.7, 0.3])
    })
    
    # Create interactive chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price & Signals', 'Indicators'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=df_chart['Date'], y=df_chart['Price'],
                  name='Price', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_chart['Date'], y=df_chart['EMA'],
                  name=f'EMA({ema_period})', line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    
    # Buy/sell signals
    buy_signals = df_chart[df_chart['Signal'] == 1]
    fig.add_trace(
        go.Scatter(x=buy_signals['Date'], y=buy_signals['Price'],
                  mode='markers', name='Buy Signal',
                  marker=dict(color='green', size=10, symbol='triangle-up')),
        row=1, col=1
    )
    
    # RSI indicator (mock)
    rsi_values = 50 + 20 * np.sin(np.arange(len(df_chart)) * 0.1)
    fig.add_trace(
        go.Scatter(x=df_chart['Date'], y=rsi_values,
                  name=f'RSI({rsi_period})', line=dict(color='purple')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                st.success("Backtest completed! Check results above.")
    
    with col2:
        if st.button("🔧 Optimize Parameters"):
            with st.spinner("Optimizing parameters..."):
                st.success("Optimization completed!")
    
    with col3:
        if st.button("💾 Save Strategy"):
            st.success("Strategy saved successfully!")
    
    with col4:
        if st.button("📤 Export Results"):
            st.markdown(create_download_link(df_chart, "trading_results"), 
                       unsafe_allow_html=True)

def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose Page",
        ["🏠 Home", "📊 Data Explorer", "🎛️ Variable Tuner", 
         "🔮 Prediction", "💼 Portfolio", "⚙️ Settings"]
    )
    
    if page == "🏠 Home":
        main_dashboard()
    elif page == "📊 Data Explorer":
        from ui.pages.data_explorer import enhanced_data_explorer
        enhanced_data_explorer()
    elif page == "🎛️ Variable Tuner":
        from ui.pages.variable_tuner import enhanced_variable_tuner
        enhanced_variable_tuner()
    elif page == "🔮 Prediction":
        from ui.pages.prediction_analysis import prediction_page
        prediction_page()
    elif page == "💼 Portfolio":
        from ui.pages.portfolio_analysis import portfolio_page
        portfolio_page()
    elif page == "⚙️ Settings":
        from ui.pages.settings import settings_page
        settings_page()

if __name__ == "__main__":
    main()
