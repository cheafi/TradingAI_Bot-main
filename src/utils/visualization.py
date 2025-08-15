"""Interactive visualization utilities using Plotly."""
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_trading_chart(
    df: pd.DataFrame,
    signals: Optional[Dict] = None,
    title: str = "Trading Analysis",
    height: int = 800,
) -> go.Figure:
    """Create an interactive candlestick chart with indicators.
    
    Args:
        df: DataFrame with OHLCV data
        signals: Optional dict of buy/sell signals
        title: Chart title
        height: Chart height in pixels
    
    Returns:
        Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, "Volume"),
        row_width=[0.7, 0.3]
    )

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC"
        ),
        row=1, col=1
    )

    # Add volume bars
    colors = ["red" if x < 0 else "green" for x in df["close"] - df["open"]]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color=colors
        ),
        row=2, col=1
    )

    # Add signals if provided
    if signals:
        for signal_type, points in signals.items():
            color = "green" if signal_type == "buy" else "red"
            fig.add_trace(
                go.Scatter(
                    x=points["timestamp"],
                    y=points["price"],
                    mode="markers",
                    name=signal_type.title(),
                    marker=dict(
                        size=8,
                        symbol="triangle-up" if signal_type == "buy" else "triangle-down",
                        color=color
                    )
                ),
                row=1, col=1
            )

    # Update layout
    fig.update_layout(
        height=height,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_portfolio_dashboard(
    portfolio_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    metrics: Dict,
) -> go.Figure:
    """Create a comprehensive portfolio dashboard.
    
    Args:
        portfolio_df: Portfolio value over time
        trades_df: Individual trades history
        metrics: Dict of performance metrics
    
    Returns:
        Plotly figure with multiple subplots
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Portfolio Value",
            "Returns Distribution",
            "Drawdown",
            "Monthly Returns",
            "Asset Allocation",
            "Trade Analysis"
        )
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["value"],
            name="Portfolio Value"
        ),
        row=1, col=1
    )
    
    # Rest of the implementation...
    return fig
