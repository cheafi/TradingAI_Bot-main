"""Enhanced Data Explorer with 3D correlations, filtering, and advanced analytics."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import yfinance as yf


def enhanced_data_explorer():
    """Enhanced data exploration page with advanced features."""
    st.header("📊 Advanced Data Explorer")
    
    # Data source selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_source = st.radio(
            "Data Source",
            ["📁 Upload CSV", "🌐 Yahoo Finance", "📝 Sample Data"],
            horizontal=True
        )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    df = None
    
    if data_source == "📁 Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload OHLCV data with Date, Open, High, Low, Close, Volume columns"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            st.success(f"Loaded {len(df)} rows of data")
    
    elif data_source == "🌐 Yahoo Finance":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Symbol", value="AAPL")
        with col2:
            period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=1)
        with col3:
            if st.button("📥 Download"):
                try:
                    with st.spinner(f"Downloading {symbol} data..."):
                        df = yf.download(symbol, period=period, progress=False)
                        st.success(f"Downloaded {len(df)} days of {symbol} data")
                except Exception as e:
                    st.error(f"Error downloading data: {e}")
    
    else:  # Sample Data
        # Generate sample OHLCV data
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq='D')
        np.random.seed(42)
        
        price_base = 100
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = price_base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        st.info("Using generated sample data")
    
    if df is not None and not df.empty:
        # Data overview
        st.subheader("📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📅 Total Days", len(df))
        with col2:
            st.metric("📊 Columns", len(df.columns))
        with col3:
            st.metric("🎯 Missing Values", df.isnull().sum().sum())
        with col4:
            if 'Close' in df.columns:
                total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                st.metric("📈 Total Return", f"{total_return:.1f}%")
        
        # Data filtering
        st.subheader("🔍 Data Filtering")
        
        if 'Close' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                min_price = st.number_input(
                    "Min Price", 
                    value=float(df['Close'].min()),
                    max_value=float(df['Close'].max())
                )
            with col2:
                max_price = st.number_input(
                    "Max Price", 
                    value=float(df['Close'].max()),
                    min_value=float(df['Close'].min())
                )
            
            # Apply filters
            filtered_df = df[(df['Close'] >= min_price) & (df['Close'] <= max_price)]
            
            if len(filtered_df) != len(df):
                st.info(f"Filtered to {len(filtered_df)} rows (from {len(df)})")
                df = filtered_df
        
        # Interactive charts
        st.subheader("📈 Interactive Charts")
        
        chart_type = st.selectbox(
            "Chart Type",
            ["📊 OHLC Candlestick", "📈 Line Chart", "📉 Volume", "🔄 Returns"]
        )
        
        if chart_type == "📊 OHLC Candlestick" and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig = go.Figure(data=go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC"
            ))
            fig.update_layout(title="Candlestick Chart", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "📈 Line Chart" and 'Close' in df.columns:
            fig = px.line(df, y='Close', title="Price Line Chart")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "📉 Volume" and 'Volume' in df.columns:
            fig = px.bar(df, y='Volume', title="Volume Chart")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "🔄 Returns" and 'Close' in df.columns:
            returns = df['Close'].pct_change().dropna()
            fig = px.histogram(returns, nbins=50, title="Returns Distribution")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Correlation Analysis
        if len(df.select_dtypes(include=[np.number]).columns) >= 3:
            st.subheader("🌐 3D Correlation Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, index=0)
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols, index=1)
                with col3:
                    z_col = st.selectbox("Z-axis", numeric_cols, index=2)
                
                # Create 3D scatter plot
                fig = px.scatter_3d(
                    df.dropna(), 
                    x=x_col, 
                    y=y_col, 
                    z=z_col,
                    title=f"3D Correlation: {x_col} vs {y_col} vs {z_col}",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix heatmap
                corr_matrix = df[numeric_cols].corr()
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title="Correlation Matrix Heatmap",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe())
        
        # Export options
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download CSV"):
                csv = df.to_csv()
                st.download_button(
                    label="Download CSV file",
                    data=csv,
                    file_name="data_export.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Download JSON"):
                json_str = df.to_json(orient="records", date_format="iso")
                st.download_button(
                    label="Download JSON file",
                    data=json_str,
                    file_name="data_export.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📊 Download Summary"):
                summary = df.describe().to_csv()
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="data_summary.csv",
                    mime="text/csv"
                )
        
        # Raw data viewer (with pagination)
        st.subheader("📋 Raw Data Viewer")
        
        # Pagination
        rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        total_pages = (len(df) - 1) // rows_per_page + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
        
        st.dataframe(df.iloc[start_idx:end_idx])
        st.caption(f"Showing rows {start_idx + 1}-{end_idx} of {len(df)}")


def page():
    """Compatibility function for original interface."""
    enhanced_data_explorer()
