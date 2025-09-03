# 🤖 Multi-Agent Trading System

**Legendary Investors AI Trading Shop**

A sophisticated multi-agent trading system that implements legendary investor strategies with AI agents, featuring automatic onboarding, ROI-focused measurement, and regime-aware capital allocation.

## 🎯 System Overview

This system transforms your trading operation into a **multi-agent, multi-discipline shop** where each AI agent embodies the investment philosophy of legendary investors like Jim Simons, Ray Dalio, and Paul Tudor Jones.

### Key Features

- **🏆 Legendary Investor Agents**: AI implementations of famous investment strategies
- **⚡ Universal Agent Interface**: Standardized alpha_bps signals with automatic ensemble
- **📊 ROI-Focused Framework**: Trade only when alpha ≥ κ × (fees + slippage + borrow)
- **🎓 Automatic Onboarding**: 2-week sandbox evaluation with KPI-based promotion
- **🌍 Regime-Aware Allocation**: Dynamic capital allocation based on market conditions
- **📈 Real-Time Dashboard**: Comprehensive monitoring and control interface

## 🏗️ Architecture

```
Multi-Agent Trading System
├── 🤖 Legendary Investor Agents
│   ├── Simons StatArb Agent (PCA-based statistical arbitrage)
│   ├── Dalio Macro Agent (4-quadrant regime analysis)
│   ├── PTJ Trend Agent (Multi-timeframe momentum)
│   ├── Microstructure Agent (Execution optimization)
│   └── Drawdown Guardian (Enhanced risk management)
│
├── 🎯 Decision Gateway
│   ├── Signal Validation & Ensemble
│   ├── ROI Filtering (alpha ≥ cost threshold)
│   ├── Regime-Based Allocation
│   └── Portfolio Risk Controls
│
├── 🎓 Agent Onboarding System
│   ├── 2-Week Sandbox Evaluation
│   ├── KPI-Based Promotion/Demotion
│   ├── Performance Tracking
│   └── Capacity Scaling
│
└── 📊 Multi-Agent Dashboard
    ├── Real-Time Agent Monitoring
    ├── Portfolio Analytics
    ├── Performance Attribution
    └── System Management
```

## 🚀 Quick Start

### 1. Run the Complete System

```bash
cd /workspaces/TradingAI_Bot-main
python apps/agents/run_multi_agent_system.py
```

This will:
- Initialize all legendary investor agents
- Start the 2-week onboarding process
- Run sample trading sessions
- Show comprehensive system status

### 2. Launch the Interactive Dashboard

```bash
streamlit run apps/agents/multi_agent_dashboard.py
```

Access at: http://localhost:8501

### 3. Test Individual Components

```bash
# Test legendary investor agents
python apps/agents/legendary_investors.py

# Test decision gateway
python apps/agents/decision_gateway.py

# Test onboarding system
python apps/agents/agent_onboarding.py
```

## 🤖 Legendary Investor Agents

Each agent implements the distinctive approach of legendary investors:

### 🧮 Simons StatArb Agent
- **Strategy**: PCA-based statistical arbitrage
- **Philosophy**: Mathematical patterns in price relationships
- **Signals**: Cross-sectional mean reversion with factor neutrality
- **Preferred Regime**: Sideways markets with stable correlations

### 🌍 Dalio Macro Agent  
- **Strategy**: 4-quadrant regime analysis
- **Philosophy**: Economic cycles drive asset returns
- **Signals**: Growth/inflation regime tilts with risk parity
- **Preferred Regime**: Macro transitions and trending environments

### 📈 PTJ Trend Agent
- **Strategy**: Multi-timeframe momentum
- **Philosophy**: Trends persist longer than expected
- **Signals**: Momentum with ATR-based position sizing
- **Preferred Regime**: Strong trending markets (up or down)

### ⚡ Microstructure Agent
- **Strategy**: Execution optimization
- **Philosophy**: Implementation shortfall minimization
- **Signals**: Smart order routing and timing
- **Preferred Regime**: All regimes (execution focused)

### 🛡️ Drawdown Guardian
- **Strategy**: Enhanced risk management
- **Philosophy**: Preserve capital during adverse conditions
- **Signals**: Dynamic position sizing and stop-losses
- **Preferred Regime**: High volatility and stress periods

## 🎯 Decision Gateway

The Decision Gateway orchestrates all agents with:

### ROI-Focused Filtering
```python
# Trade only if alpha ≥ cost threshold
required_alpha = estimated_cost * min_alpha_to_cost_ratio * cost_buffer_multiplier

if signal.alpha_bps >= required_alpha:
    # Execute trade
    pass
else:
    # Reject - insufficient alpha
    pass
```

### Regime-Based Allocation
- **Low Vol Uptrend**: Focus on trend and stat-arb agents
- **High Vol Downtrend**: Emphasize macro and risk management
- **Sideways Chop**: Favor stat-arb and microstructure
- **Macro Transition**: Priority to macro and defensive agents

### Signal Ensemble Methods
- **Performance Weighting**: Based on historical edge after cost
- **Inverse Variance**: Risk-adjusted weighting
- **Bayesian Model Averaging**: Probability-weighted combination

## 🎓 Agent Onboarding Process

### 2-Week Sandbox Evaluation

1. **Week 1-2**: Agent trades with 1% allocation
2. **Evaluation Metrics**:
   - Edge after cost ≥ 3 bps
   - Information coefficient ≥ 0.05
   - Sharpe ratio ≥ 0.5
   - Max drawdown ≤ 5%

3. **Outcomes**:
   - **Pass (≥70 score)**: Promote to Active (2% allocation)
   - **Fail**: Move to Probation (0.5% allocation)

### Promotion Hierarchy

```
Candidate → Sandbox → Active → Veteran
     ↓         ↓        ↓        ↓
   0.0%     1.0%     2-15%    Up to 25%
```

### KPI-Based Management

Agents are automatically promoted/demoted based on:
- **Profitability Score** (40%): Edge after cost, win rate, total return
- **Risk Score** (30%): Sharpe ratio, drawdown, volatility control
- **Execution Score** (30%): Trade frequency, alpha consistency

## 📊 Performance Monitoring

### Real-Time Metrics
- **Alpha Generation**: Expected vs realized alpha in basis points
- **Information Coefficient**: Correlation between predicted and actual returns
- **Implementation Shortfall**: Execution cost analysis
- **Risk-Adjusted Returns**: Sharpe ratio, deflated Sharpe ratio
- **Capacity Utilization**: Current vs maximum allocation

### Attribution Analysis
- **Agent Contribution**: Individual agent P&L attribution
- **Regime Performance**: How agents perform in different market conditions
- **Factor Exposure**: Style factor analysis across the portfolio

## 🛠️ Configuration

### Risk Parameters
```python
config = DecisionGatewayConfig(
    min_alpha_to_cost_ratio=2.0,      # Minimum 2x cost coverage
    max_single_position=0.08,         # 8% max position size
    portfolio_vol_target=0.12,        # 12% annual volatility target
    max_drawdown_limit=-0.08,         # 8% max drawdown
    max_portfolio_turnover_daily=0.15  # 15% daily turnover limit
)
```

### Agent Limits
```python
config = OnboardingConfig(
    sandbox_allocation=0.01,           # 1% for new agents
    min_active_allocation=0.02,        # 2% minimum for active
    max_active_allocation=0.15,        # 15% maximum per agent
    min_edge_after_cost=3.0,          # 3 bps minimum edge
    sandbox_pass_score=70.0           # 70% to pass sandbox
)
```

## 📈 Dashboard Features

### Overview Tab
- System-wide performance metrics
- Portfolio allocation by agent
- Recent signals and executions
- Market regime indicator

### Agents Tab
- Agent status and allocations
- Performance rankings
- Individual agent analytics
- Onboarding pipeline

### Portfolio Tab
- Current positions and weights
- Sector and factor exposures
- Risk analytics and VaR
- Concentration metrics

### Performance Tab
- Cumulative returns vs benchmark
- Rolling Sharpe ratios
- Performance attribution
- Monthly returns heatmap

### System Tab
- Agent management controls
- Configuration parameters
- System health monitoring
- Audit logs

## 🔧 Advanced Features

### Regime Classification
- **Volatility Analysis**: Rolling volatility patterns
- **Trend Detection**: Multi-timeframe momentum
- **Correlation Monitoring**: Asset correlation stability
- **Macro Indicators**: Economic cycle positioning

### Cost Modeling
- **Realistic Costs**: Fees, slippage, borrowing costs
- **Market Impact**: Size and urgency adjustments
- **Timing Costs**: Delay and opportunity costs
- **Infrastructure**: Technology and data costs

### Risk Management
- **Real-Time Monitoring**: Continuous risk assessment
- **Kill Switch**: Automatic position closing on breaches
- **Stress Testing**: Portfolio performance under stress
- **Scenario Analysis**: What-if analysis tools

## 🚀 Deployment

### Development Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development server
python apps/agents/run_multi_agent_system.py
```

### Production Deployment
```bash
# Build Docker container
docker build -t multi-agent-trading .

# Deploy with docker-compose
docker-compose up -d

# Monitor with dashboard
streamlit run apps/agents/multi_agent_dashboard.py --server.port 8501
```

## 📚 Documentation

- **[Agent Development Guide](docs/agent_development.md)**: How to create new agents
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Configuration Guide](docs/configuration.md)**: System configuration options
- **[Deployment Guide](docs/deployment.md)**: Production deployment instructions

## 🤝 Contributing

### Adding New Agents

1. **Create Agent Class**: Inherit from `LegendaryAgent`
2. **Implement Strategy**: Override `generate_signals()` method
3. **Define Metadata**: Set preferred regimes and description
4. **Register Agent**: Add to `LEGENDARY_AGENTS` registry
5. **Test Integration**: Verify onboarding and signal generation

### Example New Agent
```python
class BuffettValueAgent(LegendaryAgent):
    def __init__(self):
        super().__init__(
            agent_name="buffett_value",
            description="Long-term value investing with quality focus",
            preferred_regimes=[MarketRegime.HIGH_VOL_DOWNTREND]
        )
    
    def generate_signals(self, market_data, universe, as_of):
        # Implement value investing logic
        signals = []
        # ... value screening logic ...
        return AgentOutput(signals=signals, metadata={})
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Jim Simons**: Renaissance Technologies statistical arbitrage approach
- **Ray Dalio**: Bridgewater's All Weather and regime-based investing
- **Paul Tudor Jones**: Macro trend following and risk management
- **Quantitative Finance Community**: Open source tools and methodologies

---

**Built with ❤️ for systematic traders and quantitative researchers**

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.
