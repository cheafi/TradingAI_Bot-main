# Quick Start Guide - TradingAI Pro

Get started with TradingAI Pro in 5 minutes!

## 🚀 Quick Setup

### 1. Environment Setup
```bash
# Clone repository (if not already done)
git clone https://github.com/cheafi/TradingAI_Bot-main.git
cd TradingAI_Bot-main

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Test
```bash
# Run system validation
./validate_and_deploy.sh
```

### 3. Start UI (Development Mode)
```bash
# Launch Streamlit dashboard
streamlit run ui/enhanced_dashboard.py --server.port 8501

# Open browser to: http://localhost:8501
```

## 🎯 Quick Demo

### Try the ML Pipeline
```bash
# Generate sample data and train model
python research/ml_pipeline.py --symbol AAPL --start 2020-01-01 --end 2024-01-01

# Run optimization
python research/optimize_and_backtest.py

# Analyze with Qlib workflow
python research/qlib_integration.py
```

### Test Telegram Bot (Dry Run)
```bash
# Set test token
export TELEGRAM_BOT_TOKEN="test_token_for_dry_run"

# Test bot components
python -c "
import sys
sys.path.append('.')
from src.telegram.enhanced_bot import TradingBot
print('✅ Bot imports successful')
print('✅ Ready for real token')
"
```

## 🐳 Production Quick Start

### Docker Deployment
```bash
# Start full system with monitoring
docker-compose -f docker-compose.enhanced.yml up -d

# Check services
docker-compose -f docker-compose.enhanced.yml ps

# Access services:
# - Streamlit UI: http://localhost:8501
# - Grafana: http://localhost:3000 (admin:admin)
# - Prometheus: http://localhost:9090
```

## 📊 Quick Features Demo

### 1. Streamlit Dashboard
- **Home**: System overview and key metrics
- **Data Explorer**: Interactive charts and correlations
- **Variable Tuner**: Real-time parameter adjustment
- **Prediction Analysis**: ML model insights
- **Portfolio Analysis**: Risk and performance metrics

### 2. ML Pipeline Features
- Walk-forward cross-validation
- Ensemble Random Forest models
- Economic backtesting with transaction costs
- Risk metrics (Sharpe, Kelly fraction, VaR)

### 3. Telegram Bot Commands
- `/portfolio` - Portfolio analysis
- `/chart SYMBOL` - Generate price charts
- `/suggest SYMBOL` - AI trading suggestions
- `/optimize` - Parameter optimization
- `/voice` - Voice portfolio summary

## ⚡ Quick Customization

### Modify Trading Parameters
Edit `src/config.py`:
```python
@dataclass
class Config:
    # Your custom parameters
    INITIAL_CAPITAL: float = 100_000.0
    EMA_PERIOD: int = 20
    ATR_PERIOD: int = 14
    # ... more parameters
```

### Add New Strategies
Create in `src/strategies/`:
```python
def my_strategy(df, cfg):
    # Your strategy logic
    signals = calculate_signals(df)
    return signals
```

## 🔧 Troubleshooting

### Common Issues
1. **Import errors**: `pip install -r requirements.txt`
2. **Port conflicts**: Change port in streamlit command
3. **Docker issues**: `docker system prune -f`

### Get Help
- Check [Testing Guide](../TESTING_GUIDE.md)
- See [Implementation Complete](../../IMPLEMENTATION_COMPLETE.md)
- Review error logs in terminal

## 📈 Next Steps

1. **Configure Real APIs**: Set up exchange API keys
2. **Paper Trading**: Test with simulated money
3. **Customize Strategies**: Modify trading logic
4. **Monitor Performance**: Use Grafana dashboards
5. **Scale Up**: Deploy to cloud infrastructure

**Ready to trade smarter with AI! 🚀**
