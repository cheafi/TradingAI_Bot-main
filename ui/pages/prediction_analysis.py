"""Prediction Analysis page with ML model insights."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def prediction_page():
    """ML prediction analysis and model insights."""
    st.header("🔮 Prediction Analysis")
    
    st.subheader("📊 Model Performance")
    
    # Mock model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", "72.5%", "+2.1%")
    with col2:
        st.metric("📈 Precision", "68.3%", "+1.8%")
    with col3:
        st.metric("🔄 Recall", "71.2%", "-0.5%")
    with col4:
        st.metric("⚡ F1-Score", "69.7%", "+1.2%")
    
    # Feature importance
    st.subheader("🎲 Feature Importance")
    
    features = ["RSI_14", "ATR_14", "EMA_20", "Volume", "Price_Change", 
                "Volatility", "Momentum", "Support_Resistance"]
    importance = np.random.dirichlet(np.ones(len(features))) * 100
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance (%)",
        color=importance,
        color_continuous_scale="viridis"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction confidence
    st.subheader("📊 Prediction Confidence Distribution")
    
    confidence_scores = np.random.beta(2, 2, 1000)
    fig = px.histogram(
        confidence_scores,
        nbins=30,
        title="Model Confidence Distribution",
        labels={'value': 'Confidence Score', 'count': 'Frequency'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    st.subheader("📋 Recent Predictions")
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='H')
    predictions_df = pd.DataFrame({
        'Timestamp': dates,
        'Symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], 10),
        'Prediction': np.random.choice(['BUY', 'SELL', 'HOLD'], 10),
        'Confidence': np.random.uniform(0.5, 0.95, 10),
        'Price': np.random.uniform(100, 200, 10)
    })
    
    st.dataframe(predictions_df, use_container_width=True)
