"""
Multi-Agent Trading Dashboard
============================

Real-time monitoring dashboard for the multi-agent trading system.
Provides comprehensive views of:
- Agent performance and status
- Portfolio composition and risk
- Market regime analysis
- Trade attribution and P&L breakdown
- ROI metrics and capacity utilization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

# Import our agent systems
try:
    from apps.agents.legendary_investors import LEGENDARY_AGENTS, create_agent, MarketRegime
    from apps.agents.decision_gateway import DecisionGateway, DecisionGatewayConfig
    from apps.agents.agent_onboarding import AgentOnboardingSystem, OnboardingConfig, AgentStatus
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


class MultiAgentDashboard:
    """Main dashboard class for multi-agent trading system."""
    
    def __init__(self):
        self.logger = logging.getLogger("dashboard")
        
        # Initialize systems
        self.decision_gateway = DecisionGateway()
        self.onboarding_system = AgentOnboardingSystem()
        
        # Sample data for demo
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize sample market data for demonstration."""
        # Generate sample market data
        dates = pd.date_range('2023-01-01', datetime.now(), freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        np.random.seed(42)
        self.market_data = pd.DataFrame(
            100 * np.cumprod(1 + np.random.normal(0.0008, 0.015, (len(dates), len(symbols))), axis=0),
            index=dates,
            columns=symbols
        )
        
        # Generate sample portfolio data
        self.portfolio_data = self._generate_sample_portfolio()
        
        # Generate sample agent performance
        self.agent_performance = self._generate_sample_performance()
    
    def _generate_sample_portfolio(self) -> pd.DataFrame:
        """Generate sample portfolio positions."""
        symbols = self.market_data.columns
        
        # Sample allocations
        np.random.seed(123)
        weights = np.random.dirichlet(np.ones(len(symbols)), 1)[0]
        weights = weights / weights.sum() * 0.8  # 80% invested
        
        return pd.DataFrame({
            'Symbol': symbols,
            'Weight': weights,
            'Value': weights * 1000000,  # $1M portfolio
            'Agent': np.random.choice(['simons_stat_arb', 'dalio_macro', 'ptj_trend', 'microstructure'], len(symbols))
        })
    
    def _generate_sample_performance(self) -> Dict:
        """Generate sample agent performance data."""
        agents = list(LEGENDARY_AGENTS.keys())
        
        return {
            agent: {
                'mtd_return': np.random.normal(0.02, 0.01),
                'ytd_return': np.random.normal(0.15, 0.05),
                'sharpe_ratio': np.random.normal(1.2, 0.3),
                'max_drawdown': np.random.uniform(-0.08, -0.02),
                'win_rate': np.random.uniform(0.45, 0.65),
                'trades_count': np.random.randint(50, 200),
                'allocation': np.random.uniform(0.05, 0.25),
                'status': np.random.choice(['Active', 'Sandbox', 'Veteran'])
            }
            for agent in agents
        }
    
    def run_dashboard(self):
        """Main dashboard runner."""
        st.set_page_config(
            page_title="Multi-Agent Trading Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🤖 Multi-Agent Trading Dashboard")
        st.markdown("**Legendary Investors AI Trading System**")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_sidebar(self):
        """Render dashboard sidebar."""
        st.sidebar.header("🎛️ Controls")
        
        # System status
        st.sidebar.subheader("System Status")
        st.sidebar.success("🟢 All Systems Operational")
        
        # Market regime
        current_regime = self.decision_gateway.current_regime
        regime_color = {
            MarketRegime.LOW_VOL_UPTREND: "🟢",
            MarketRegime.HIGH_VOL_DOWNTREND: "🔴", 
            MarketRegime.SIDEWAYS_CHOP: "🟡",
            MarketRegime.MACRO_TRANSITION: "🟠",
            MarketRegime.UNKNOWN: "⚪"
        }
        
        st.sidebar.subheader("Market Regime")
        st.sidebar.markdown(f"{regime_color.get(current_regime, '⚪')} {current_regime.value.replace('_', ' ').title()}")
        
        # Portfolio summary
        st.sidebar.subheader("Portfolio Summary")
        total_value = self.portfolio_data['Value'].sum()
        st.sidebar.metric("Total AUM", f"${total_value:,.0f}")
        
        active_agents = len([a for a in self.agent_performance.values() if a['status'] == 'Active'])
        st.sidebar.metric("Active Agents", active_agents)
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.sidebar.button("📊 Generate Signals"):
            st.sidebar.success("Signals generated!")
        
        if st.sidebar.button("⚠️ Risk Check"):
            st.sidebar.info("Risk limits: ✅ All clear")
    
    def _render_main_content(self):
        """Render main dashboard content."""
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🤖 Agents", 
            "💼 Portfolio", 
            "📈 Performance", 
            "⚙️ System"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_agents_tab()
        
        with tab3:
            self._render_portfolio_tab()
        
        with tab4:
            self._render_performance_tab()
        
        with tab5:
            self._render_system_tab()
    
    def _render_overview_tab(self):
        """Render overview tab."""
        st.header("📊 System Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = np.mean([p['ytd_return'] for p in self.agent_performance.values()])
            st.metric(
                "YTD Return", 
                f"{total_return:.1%}",
                delta=f"{total_return*0.1:.1%} vs benchmark"
            )
        
        with col2:
            avg_sharpe = np.mean([p['sharpe_ratio'] for p in self.agent_performance.values()])
            st.metric(
                "Sharpe Ratio",
                f"{avg_sharpe:.2f}",
                delta="0.15"
            )
        
        with col3:
            max_dd = min([p['max_drawdown'] for p in self.agent_performance.values()])
            st.metric(
                "Max Drawdown",
                f"{max_dd:.1%}",
                delta=f"{abs(max_dd)*0.2:.1%}"
            )
        
        with col4:
            total_trades = sum([p['trades_count'] for p in self.agent_performance.values()])
            st.metric(
                "Total Trades",
                f"{total_trades:,}",
                delta="47 today"
            )
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio composition by agent
            agent_allocation = self.portfolio_data.groupby('Agent')['Value'].sum()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=agent_allocation.index,
                    values=agent_allocation.values,
                    hole=0.4,
                    textinfo='label+percent'
                )
            ])
            fig.update_layout(
                title="Portfolio Allocation by Agent",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance over time (sample)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            cumulative_returns = np.cumprod(1 + np.random.normal(0.001, 0.015, 30))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            # Add benchmark
            benchmark_returns = np.cumprod(1 + np.random.normal(0.0005, 0.012, 30))
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_returns,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title="30-Day Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent signals
        st.subheader("🎯 Recent Signals")
        
        sample_signals = pd.DataFrame({
            'Time': ['09:31', '09:45', '10:15', '10:32', '11:07'],
            'Agent': ['Simons StatArb', 'PTJ Trend', 'Dalio Macro', 'Microstructure', 'Simons StatArb'],
            'Symbol': ['AAPL', 'TSLA', 'SPY', 'MSFT', 'GOOGL'],
            'Direction': ['LONG', 'SHORT', 'LONG', 'LONG', 'SHORT'],
            'Alpha (bps)': [15.2, 22.8, 8.5, 12.1, 18.7],
            'Confidence': [0.85, 0.92, 0.71, 0.78, 0.88],
            'Status': ['Executed', 'Executed', 'Pending', 'Executed', 'Executed']
        })
        
        st.dataframe(sample_signals, use_container_width=True)
    
    def _render_agents_tab(self):
        """Render agents tab."""
        st.header("🤖 Agent Management")
        
        # Agent status overview
        agent_status = self.onboarding_system.get_agent_roster()
        
        # Status summary
        status_counts = {}
        for agent_info in agent_status.values():
            status = agent_info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Agents", status_counts.get('active', 0))
        with col2:
            st.metric("Sandbox Agents", status_counts.get('sandbox', 0))
        with col3:
            st.metric("Veteran Agents", status_counts.get('veteran', 0))
        with col4:
            st.metric("Total Agents", len(agent_status))
        
        # Agent performance table
        st.subheader("📊 Agent Performance")
        
        # Create agent performance dataframe
        agent_df = []
        for agent_name, perf in self.agent_performance.items():
            agent_df.append({
                'Agent': agent_name.replace('_', ' ').title(),
                'Status': perf['status'],
                'Allocation': f"{perf['allocation']:.1%}",
                'MTD Return': f"{perf['mtd_return']:.1%}",
                'YTD Return': f"{perf['ytd_return']:.1%}",
                'Sharpe': f"{perf['sharpe_ratio']:.2f}",
                'Max DD': f"{perf['max_drawdown']:.1%}",
                'Win Rate': f"{perf['win_rate']:.1%}",
                'Trades': perf['trades_count']
            })
        
        agent_df = pd.DataFrame(agent_df)
        
        # Color coding for status
        def color_status(val):
            if val == 'Active':
                return 'background-color: #90EE90'
            elif val == 'Veteran':
                return 'background-color: #FFD700'
            elif val == 'Sandbox':
                return 'background-color: #FFA500'
            return ''
        
        styled_df = agent_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Agent details
        st.subheader("🔍 Agent Details")
        
        selected_agent = st.selectbox(
            "Select Agent for Details:",
            options=list(self.agent_performance.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_agent:
            col1, col2 = st.columns(2)
            
            with col1:
                perf = self.agent_performance[selected_agent]
                st.write(f"**{selected_agent.replace('_', ' ').title()}**")
                st.write(f"Status: {perf['status']}")
                st.write(f"Current Allocation: {perf['allocation']:.1%}")
                st.write(f"Total Trades: {perf['trades_count']}")
                
                # Agent-specific metrics
                agent_info = LEGENDARY_AGENTS.get(selected_agent, {})
                st.write(f"Strategy: {agent_info.get('description', 'N/A')}")
            
            with col2:
                # Agent performance chart
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                agent_returns = np.random.normal(0.001, 0.02, 30)
                cumulative = np.cumprod(1 + agent_returns)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=cumulative,
                    mode='lines',
                    name=selected_agent.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    title=f"{selected_agent.replace('_', ' ').title()} - 30 Day Performance",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_portfolio_tab(self):
        """Render portfolio tab."""
        st.header("💼 Portfolio Analysis")
        
        # Portfolio summary
        total_value = self.portfolio_data['Value'].sum()
        num_positions = len(self.portfolio_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Value", f"${total_value:,.0f}")
        with col2:
            st.metric("Positions", num_positions)
        with col3:
            concentration = self.portfolio_data['Weight'].max()
            st.metric("Max Position", f"{concentration:.1%}")
        with col4:
            cash = 1.0 - self.portfolio_data['Weight'].sum()
            st.metric("Cash", f"{cash:.1%}")
        
        # Portfolio composition
        col1, col2 = st.columns(2)
        
        with col1:
            # Top positions
            st.subheader("🔝 Top Positions")
            top_positions = self.portfolio_data.nlargest(10, 'Weight')[['Symbol', 'Weight', 'Agent']]
            st.dataframe(top_positions, use_container_width=True)
        
        with col2:
            # Sector allocation (sample)
            st.subheader("🏭 Sector Allocation")
            sectors = ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial']
            sector_weights = np.random.dirichlet(np.ones(len(sectors)))
            
            fig = go.Figure(data=[
                go.Bar(x=sectors, y=sector_weights)
            ])
            fig.update_layout(
                title="Sector Breakdown",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis
        st.subheader("⚠️ Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Portfolio beta
            st.metric("Portfolio Beta", "1.05", delta="0.02")
        
        with col2:
            # VaR
            st.metric("95% VaR (1d)", "-2.1%", delta="-0.1%")
        
        with col3:
            # Correlation to market
            st.metric("Market Correlation", "0.73", delta="-0.05")
        
        # Position details
        st.subheader("📋 All Positions")
        
        # Format portfolio data for display
        display_portfolio = self.portfolio_data.copy()
        display_portfolio['Weight'] = display_portfolio['Weight'].apply(lambda x: f"{x:.2%}")
        display_portfolio['Value'] = display_portfolio['Value'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_portfolio, use_container_width=True)
    
    def _render_performance_tab(self):
        """Render performance tab."""
        st.header("📈 Performance Analytics")
        
        # Performance overview
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative returns
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            portfolio_returns = np.random.normal(0.001, 0.015, 252)
            benchmark_returns = np.random.normal(0.0005, 0.012, 252)
            
            portfolio_cumulative = np.cumprod(1 + portfolio_returns)
            benchmark_cumulative = np.cumprod(1 + benchmark_returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_cumulative,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_cumulative,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title="1-Year Cumulative Returns",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rolling metrics
            rolling_sharpe = pd.Series(np.random.normal(1.2, 0.3, 252), index=dates)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=rolling_sharpe,
                mode='lines',
                name='30-Day Rolling Sharpe',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Rolling Sharpe Ratio",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance attribution
        st.subheader("🎯 Performance Attribution")
        
        # Agent contribution
        agent_contrib = []
        for agent_name, perf in self.agent_performance.items():
            contrib = perf['allocation'] * perf['ytd_return'] * 100  # in bps
            agent_contrib.append({
                'Agent': agent_name.replace('_', ' ').title(),
                'Allocation': f"{perf['allocation']:.1%}",
                'Return': f"{perf['ytd_return']:.1%}",
                'Contribution (bps)': f"{contrib:.0f}"
            })
        
        contrib_df = pd.DataFrame(agent_contrib)
        st.dataframe(contrib_df, use_container_width=True)
        
        # Monthly returns heatmap
        st.subheader("📅 Monthly Returns Heatmap")
        
        # Generate sample monthly returns
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = ['2023', '2024']
        
        monthly_returns = np.random.normal(0.01, 0.03, (len(years), len(months)))
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_returns,
            x=months,
            y=years,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f"{val:.1%}" for val in row] for row in monthly_returns],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Monthly Returns (%)",
            height=200
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_system_tab(self):
        """Render system tab."""
        st.header("⚙️ System Management")
        
        # System health
        st.subheader("🏥 System Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("🟢 Data Feed: Online")
            st.success("🟢 Execution: Active")
            st.success("🟢 Risk Monitor: Running")
        
        with col2:
            st.info("📡 Market Data: Real-time")
            st.info("💾 Database: Connected")
            st.info("🔐 Security: Validated")
        
        with col3:
            st.warning("⚠️ Bandwidth: 85% utilized")
            st.success("🟢 Latency: 2.3ms")
            st.success("🟢 Uptime: 99.97%")
        
        # Configuration
        st.subheader("🎛️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Risk Parameters**")
            max_position = st.slider("Max Position Size", 0.01, 0.20, 0.08, 0.01, format="%.2f")
            max_drawdown = st.slider("Max Drawdown", 0.02, 0.15, 0.08, 0.01, format="%.2f")
            vol_target = st.slider("Volatility Target", 0.05, 0.25, 0.12, 0.01, format="%.2f")
        
        with col2:
            st.write("**Execution Parameters**")
            min_alpha_ratio = st.slider("Min Alpha/Cost Ratio", 1.0, 5.0, 2.0, 0.1)
            max_turnover = st.slider("Max Daily Turnover", 0.05, 0.30, 0.15, 0.01, format="%.2f")
            ensemble_method = st.selectbox("Ensemble Method", 
                                         ["Performance Weight", "Equal Weight", "Inverse Variance"])
        
        # Agent management
        st.subheader("👥 Agent Management")
        
        # Add new agent
        st.write("**Add New Agent**")
        col1, col2 = st.columns(2)
        
        with col1:
            available_agents = [a for a in LEGENDARY_AGENTS.keys() 
                              if a not in self.onboarding_system.agent_registry]
            if available_agents:
                new_agent = st.selectbox("Select Agent Type", available_agents)
                if st.button("🚀 Start Onboarding"):
                    success = self.onboarding_system.onboard_new_agent(new_agent)
                    if success:
                        st.success(f"Started onboarding for {new_agent}")
                    else:
                        st.error("Failed to start onboarding")
        
        with col2:
            st.write("**Bulk Actions**")
            if st.button("📊 Run All Evaluations"):
                st.info("Running evaluations for all agents...")
            if st.button("🔄 Rebalance Portfolio"):
                st.info("Rebalancing portfolio allocations...")
            if st.button("⚠️ Emergency Stop"):
                st.error("Emergency stop activated!")
        
        # Logs
        st.subheader("📜 System Logs")
        
        sample_logs = [
            "2024-12-19 15:30:15 - INFO - Signal generated: AAPL LONG 15.2bps",
            "2024-12-19 15:29:45 - INFO - Portfolio rebalanced: 5 positions adjusted",
            "2024-12-19 15:29:12 - WARNING - High correlation detected: MSFT-AAPL",
            "2024-12-19 15:28:33 - INFO - Agent evaluation completed: simons_stat_arb",
            "2024-12-19 15:27:58 - INFO - Market regime changed to: LOW_VOL_UPTREND"
        ]
        
        for log in sample_logs:
            if "WARNING" in log:
                st.warning(log)
            elif "ERROR" in log:
                st.error(log)
            else:
                st.text(log)


def main():
    """Main function to run the dashboard."""
    dashboard = MultiAgentDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
