import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import time
import asyncio
import pandas as pd
from src.utils.data import synthetic_ohlcv, fetch_ohlcv_ccxt
from src.strategies.scalping import enrich, signals
from src.config import cfg
from src.utils.news import get_realtime_news_and_sentiment
from src.core.agent_manager import AgentManager
from src.portfolio.optimizer import PortfolioOptimizer
from src.utils.risk import RiskManager
from src.utils.sustainability import get_trade_carbon_footprint
from src.backtesting.backtester import Backtester
from src.strategies.sample_strategy import SampleStrategy
from src.ml.price_prediction import train_lstm_model, predict_price

st.title("Trading AI Dashboard")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Select Mode", ["demo", "live"])
    symbol = st.text_input("Symbol", "AAPL")
    limit = st.slider("Data Limit", 10, 100, 50)

# Main content
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["Market Data", "Strategy Tester", "Backtest Results", "Risk Management", "Portfolio Optimization", "Agent Signals", "Sustainability and Ethics", "Backtesting", "Price Prediction"])

with tab1:
    st.header("Market Data")
    try:
        if mode == "demo":
            df = synthetic_ohlcv(symbol, limit)
        else:
            df = fetch_ohlcv_ccxt(symbol, limit=limit)

        # Use Plotly for interactive charts
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ))
        fig.update_layout(title=f'{symbol} Price Action')
        st.plotly_chart(fig, use_container_width=True)

        # Fetch and display real-time news and sentiment
        news_api_key = st.secrets["NEWS_API_KEY"]  # Access API key from Streamlit secrets
        if news_api_key:
            realtime_data = asyncio.run(get_realtime_news_and_sentiment(symbol, news_api_key))
            if realtime_data:
                st.subheader("Real-Time News")
                for article in realtime_data["news"]:
                    st.write(f"**{article['title']}**")
                    st.write(f"{article['description']}")
                    st.write(f"[Read More]({article['url']})")
                st.subheader(f"Average Sentiment: {realtime_data['avg_sentiment']:.2f}")
            else:
                st.write("No news available.")
        else:
            st.warning("News API key not found. Please add it to Streamlit secrets.")

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

with tab2:
    st.header("Strategy Tester")
    # Strategy testing code here

with tab3:
    st.header("Backtest Results")
    # Backtest results code here

with tab4:
    st.header("Risk Management")
    risk_manager = RiskManager()
    metrics = risk_manager.calculate_risk_metrics(returns_df)

    st.metric("Value at Risk (95%)", f"{metrics.var_95:.2%}")
    st.metric("Conditional Value at Risk (95%)", f"{metrics.cvar_95:.2%}")
    st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")

    shock = st.slider("Stress Test Shock", -0.20, 0.0, -0.10, 0.01)
    shocked_return = risk_manager.stress_test(returns_df.iloc[:, 0], shock)
    st.metric("Shock Test Return", f"{shocked_return:.2%}")

    st.subheader("Correlation Matrix")
    st.dataframe(metrics.correlation_matrix)

with tab5:
    st.header("Portfolio Optimization")
    optimizer = PortfolioOptimizer()

    target_return = st.slider("Target Annual Return", 0.0, 0.5, 0.2, 0.05)
    weights = optimizer.optimize(returns_df, target_return)

    st.subheader("Optimal Portfolio Weights")
    st.bar_chart(pd.Series(weights))

with tab6:
    st.header("Agent Signals")
    agent_manager = AgentManager(cfg.symbols)
    agent_signals = agent_manager.generate_signals()
    consolidated_signals = agent_manager.consolidate_signals(agent_signals)

    st.subheader("Agent Signals")
    st.write(agent_signals)
    st.subheader("Consolidated Signals")
    st.write(consolidated_signals)

with tab7:
    st.header("Sustainability and Ethics")
    trade_volume = 1000  # Example trade volume
    region = "US"  # Example region
    carbon_api_key = st.secrets["CARBON_API_KEY"]  # Access API key from Streamlit secrets

    if carbon_api_key:
        carbon_footprint = asyncio.run(get_trade_carbon_footprint(trade_volume, region, carbon_api_key))
        if carbon_footprint is not None:
            st.metric("Trade Carbon Footprint", f"{carbon_footprint:.4f} kg CO2e")
        else:
            st.write("Could not calculate carbon footprint.")
    else:
        st.warning("Carbon API key not found. Please add it to Streamlit secrets.")

    st.subheader("Bias Detection")
    agent_manager = AgentManager(cfg.symbols)
    agent_signals = agent_manager.generate_signals()
    bias_metrics = agent_manager.detect_bias(agent_signals)
    st.write(bias_metrics)

with tab8:
    st.header("Backtesting")
    if mode == "demo":
        df = synthetic_ohlcv(symbol, limit)
    else:
        df = fetch_ohlcv_ccxt(symbol, limit=limit)

    backtester = Backtester(df, SampleStrategy)
    pnl, portfolio_value = backtester.run()

    st.write(f"Profit/Loss: {pnl:.2f}")
    st.write(f"Final Portfolio Value: {portfolio_value:.2f}")

with tab9:
    st.header("Price Prediction")
    if mode == "demo":
        df = synthetic_ohlcv(symbol, limit)
    else:
        df = fetch_ohlcv_ccxt(symbol, limit=limit)

    model, scaler = train_lstm_model(df)
    if model is not None and scaler is not None:
        predicted_price = predict_price(model, df, scaler)
        if predicted_price is not None:
            st.write(f"Predicted Price: {predicted_price:.2f}")
        else:
            st.write("Could not predict price.")
    else:
        st.write("Could not train price prediction model.")