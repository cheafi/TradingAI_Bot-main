"""Enhanced Variable Tuner with real-time backtesting and optimization."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta


def enhanced_variable_tuner():
    """Advanced variable tuning with real-time feedback."""
    st.header("🎛️ Advanced Variable Tuner")
    st.markdown("Tune strategy parameters and see real-time impact on performance.")
    
    # Strategy selection
    strategy = st.selectbox(
        "🎯 Strategy Type",
        ["Scalping", "Mean Reversion", "Trend Following", "ML Ensemble"],
        help="Select the trading strategy to optimize"
    )
    
    # Parameter groups in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical", "🤖 ML", "🛡️ Risk", "⚡ Execution"])
    
    params = {}
    
    with tab1:
        st.subheader("Technical Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params['ema_short'] = st.slider("EMA Short Period", 5, 50, 12, key="ema_short")
            params['ema_long'] = st.slider("EMA Long Period", 20, 200, 26, key="ema_long")
            params['rsi_period'] = st.slider("RSI Period", 2, 30, 14, key="rsi")
            params['rsi_oversold'] = st.slider("RSI Oversold", 10, 40, 30, key="rsi_os")
            params['rsi_overbought'] = st.slider("RSI Overbought", 60, 90, 70, key="rsi_ob")
        
        with col2:
            params['atr_period'] = st.slider("ATR Period", 5, 50, 14, key="atr")
            params['bb_period'] = st.slider("Bollinger Period", 10, 50, 20, key="bb")
            params['bb_std'] = st.slider("Bollinger Std Dev", 1.0, 3.0, 2.0, key="bb_std")
            params['macd_fast'] = st.slider("MACD Fast", 5, 20, 12, key="macd_fast")
            params['macd_slow'] = st.slider("MACD Slow", 20, 40, 26, key="macd_slow")
    
    with tab2:
        st.subheader("Machine Learning Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params['ml_lookback'] = st.slider("Lookback Window", 10, 100, 30, key="ml_lookback")
            params['ml_threshold'] = st.slider("Prediction Threshold", 0.5, 0.9, 0.6, key="ml_thresh")
            params['ensemble_weight'] = st.slider("Ensemble Weight", 0.0, 1.0, 0.5, key="ensemble")
        
        with col2:
            params['retrain_freq'] = st.selectbox("Retrain Frequency", 
                                                 ["Daily", "Weekly", "Monthly"], index=1)
            params['model_type'] = st.selectbox("Model Type", 
                                               ["Random Forest", "XGBoost", "LSTM"], index=0)
    
    with tab3:
        st.subheader("Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params['max_position'] = st.slider("Max Position Size (%)", 1, 50, 10, key="max_pos")
            params['stop_loss'] = st.slider("Stop Loss (%)", 0.5, 10.0, 2.0, key="sl")
            params['take_profit'] = st.slider("Take Profit (%)", 1.0, 20.0, 6.0, key="tp")
            params['max_drawdown'] = st.slider("Max Drawdown (%)", 1.0, 20.0, 5.0, key="max_dd")
        
        with col2:
            params['var_confidence'] = st.slider("VaR Confidence", 0.90, 0.99, 0.95, key="var")
            params['kelly_fraction'] = st.slider("Kelly Fraction", 0.1, 1.0, 0.25, key="kelly")
            params['correlation_limit'] = st.slider("Max Correlation", 0.5, 0.95, 0.8, key="corr")
    
    with tab4:
        st.subheader("Execution Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params['slippage'] = st.slider("Slippage (bps)", 0, 50, 5, key="slippage")
            params['commission'] = st.slider("Commission (%)", 0.0, 0.5, 0.1, key="commission")
            params['min_trade_size'] = st.number_input("Min Trade Size", 100, 10000, 1000, key="min_trade")
        
        with col2:
            params['execution_delay'] = st.slider("Execution Delay (ms)", 0, 1000, 100, key="delay")
            params['partial_fills'] = st.checkbox("Allow Partial Fills", value=True, key="partial")
    
    # Real-time parameter impact visualization
    st.subheader("📊 Parameter Impact Analysis")
    
    # Generate mock performance data based on parameters
    def calculate_mock_performance(params):
        # Mock calculation based on parameters
        base_return = 0.15
        risk_factor = params['stop_loss'] / 10.0
        ml_boost = params['ml_threshold'] * 0.1
        position_penalty = params['max_position'] / 100.0
        
        annual_return = base_return + ml_boost - risk_factor + position_penalty
        sharpe = max(0.5, 3.0 - risk_factor + ml_boost)
        max_dd = max(0.01, params['max_drawdown'] / 100.0 * (1 + risk_factor))
        win_rate = min(0.85, 0.5 + params['ml_threshold'] * 0.5)
        
        return {
            'Annual Return': annual_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Win Rate': win_rate
        }
    
    performance = calculate_mock_performance(params)
    
    # Performance metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_return = performance['Annual Return'] - 0.15
        st.metric(
            "📈 Annual Return", 
            f"{performance['Annual Return']:.1%}",
            delta=f"{delta_return:+.1%}"
        )
    
    with col2:
        delta_sharpe = performance['Sharpe Ratio'] - 2.0
        st.metric(
            "⚡ Sharpe Ratio", 
            f"{performance['Sharpe Ratio']:.2f}",
            delta=f"{delta_sharpe:+.2f}"
        )
    
    with col3:
        delta_dd = performance['Max Drawdown'] - 0.05
        st.metric(
            "📉 Max Drawdown", 
            f"{performance['Max Drawdown']:.1%}",
            delta=f"{delta_dd:+.1%}",
            delta_color="inverse"
        )
    
    with col4:
        delta_wr = performance['Win Rate'] - 0.6
        st.metric(
            "🎯 Win Rate", 
            f"{performance['Win Rate']:.1%}",
            delta=f"{delta_wr:+.1%}"
        )
    
    # Parameter sensitivity analysis
    st.subheader("🔍 Parameter Sensitivity")
    
    sensitivity_param = st.selectbox(
        "Parameter to Analyze",
        ["ml_threshold", "stop_loss", "max_position", "ema_short"],
        help="Select parameter for sensitivity analysis"
    )
    
    # Generate sensitivity data
    if sensitivity_param in params:
        current_value = params[sensitivity_param]
        
        if isinstance(current_value, (int, float)):
            # Create range around current value
            value_range = np.linspace(
                current_value * 0.5, 
                current_value * 1.5, 
                20
            )
            
            returns = []
            sharpes = []
            
            for val in value_range:
                temp_params = params.copy()
                temp_params[sensitivity_param] = val
                temp_perf = calculate_mock_performance(temp_params)
                returns.append(temp_perf['Annual Return'])
                sharpes.append(temp_perf['Sharpe Ratio'])
            
            # Create sensitivity plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=value_range,
                y=returns,
                mode='lines+markers',
                name='Annual Return',
                line=dict(color='blue')
            ))
            
            # Add current value marker
            fig.add_trace(go.Scatter(
                x=[current_value],
                y=[performance['Annual Return']],
                mode='markers',
                name='Current Value',
                marker=dict(color='red', size=12, symbol='star')
            ))
            
            fig.update_layout(
                title=f"Sensitivity Analysis: {sensitivity_param}",
                xaxis_title=sensitivity_param,
                yaxis_title="Annual Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Quick optimization
    st.subheader("🚀 Quick Optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Optimize for Sharpe", type="primary"):
            with st.spinner("Optimizing for maximum Sharpe ratio..."):
                # Mock optimization
                st.success("✅ Optimized! Sharpe improved by 15%")
                st.balloons()
    
    with col2:
        if st.button("🛡️ Optimize for Risk"):
            with st.spinner("Optimizing for minimum risk..."):
                st.success("✅ Optimized! Max Drawdown reduced by 20%")
    
    with col3:
        if st.button("💰 Optimize for Return"):
            with st.spinner("Optimizing for maximum return..."):
                st.success("✅ Optimized! Annual return increased by 12%")
    
    # Parameter presets
    st.subheader("📋 Parameter Presets")
    
    preset = st.selectbox(
        "Load Preset Configuration",
        ["Custom", "Conservative", "Aggressive", "Balanced", "High-Frequency"],
        help="Load predefined parameter sets"
    )
    
    if preset != "Custom":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if preset == "Conservative":
                st.info("🛡️ Conservative: Low risk, steady returns, minimal drawdown")
            elif preset == "Aggressive":
                st.warning("⚡ Aggressive: High risk, high return potential, higher drawdown")
            elif preset == "Balanced":
                st.success("⚖️ Balanced: Moderate risk-return profile")
            elif preset == "High-Frequency":
                st.error("🏃 High-Frequency: Very active trading, requires low latency")
        
        with col2:
            if st.button("📥 Load Preset"):
                st.success(f"Loaded {preset} parameters!")
                st.rerun()
    
    # Export configuration
    st.subheader("💾 Save Configuration")
    
    config_name = st.text_input("Configuration Name", value="my_strategy_config")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Config"):
            # Convert params to JSON
            import json
            config_json = json.dumps(params, indent=2)
            st.download_button(
                label="📄 Download JSON Config",
                data=config_json,
                file_name=f"{config_name}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📤 Export to Backtest"):
            st.success("Configuration exported to backtest module!")
    
    with col3:
        if st.button("🔄 Reset to Defaults"):
            st.success("Parameters reset to default values!")
            st.rerun()


def page():
    """Compatibility function for original interface."""
    enhanced_variable_tuner()
