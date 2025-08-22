"""Portfolio Analysis page."""
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def portfolio_page():
    """Portfolio analysis and optimization page."""
    st.header("💼 Portfolio Analysis")
    
    # Portfolio overview
    st.subheader("📊 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Value", "$125,432", "+$2,341")
    with col2:
        st.metric("📈 Total Return", "12.4%", "+0.8%")
    with col3:
        st.metric("⚡ Sharpe Ratio", "2.1", "+0.1")
    with col4:
        st.metric("📉 Max Drawdown", "3.2%", "-0.5%")
    
    # Asset allocation
    st.subheader("🥧 Asset Allocation")
    
    assets = ["AAPL", "MSFT", "GOOGL", "TSLA", "BTC", "ETH", "Cash"]
    weights = np.random.dirichlet(np.ones(len(assets)))
    
    fig = px.pie(
        values=weights,
        names=assets,
        title="Current Portfolio Allocation"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance chart
    st.subheader("📈 Portfolio Performance")
    
    dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
    portfolio_value = 100000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
    
    fig = px.line(
        x=dates,
        y=portfolio_value,
        title="Portfolio Value Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
