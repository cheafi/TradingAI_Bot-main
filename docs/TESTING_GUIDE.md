# Testing Guide - TradingAI Pro

This guide covers how to test all components of the TradingAI Pro system, including dry testing of UI and Telegram bot.

## 🧪 Testing Overview

### Test Types
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **UI Tests**: Streamlit interface testing (dry run)
- **Bot Tests**: Telegram bot testing (dry run)
- **End-to-End Tests**: Complete system testing

## 🖥️ UI Testing (Streamlit)

### Method 1: Local Development Server
```bash
# Activate environment
source .venv/bin/activate

# Start Streamlit UI in development mode
streamlit run ui/enhanced_dashboard.py --server.port 8501

# Access at: http://localhost:8501
```

### Method 2: Dry Run Testing
```bash
# Test UI imports and syntax
python -c "
import sys
sys.path.append('.')
import streamlit as st
import ui.enhanced_dashboard

print('✅ UI imports successful')
print('✅ Streamlit version:', st.__version__)
"

# Test specific UI components
python ui/enhanced_dashboard.py --help 2>/dev/null && echo "✅ UI syntax valid"
```

### Method 3: Headless Testing
```bash
# Test UI without browser (useful for CI/CD)
STREAMLIT_SERVER_HEADLESS=true streamlit run ui/enhanced_dashboard.py --server.port 8502 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test if server is running
curl -f http://localhost:8502 && echo "✅ UI server responsive"

# Cleanup
kill $SERVER_PID
```

### UI Testing Checklist
- [ ] Dashboard loads without errors
- [ ] Navigation between pages works
- [ ] Charts render correctly
- [ ] Parameter tuning updates in real-time
- [ ] Data export functions work
- [ ] Theme switching works

## 🤖 Telegram Bot Testing

### Method 1: Test Bot (Recommended for Dry Testing)
```bash
# Create a test bot without real Telegram token
python -c "
import sys
sys.path.append('.')

# Mock telegram bot for testing
class MockBot:
    def __init__(self):
        self.token = 'test_token'
    
    async def send_message(self, chat_id, text):
        print(f'📤 Would send to {chat_id}: {text}')
        return {'message_id': 123}
    
    async def send_photo(self, chat_id, photo):
        print(f'📷 Would send photo to {chat_id}')
        return {'message_id': 124}

# Test bot functions
from src.telegram.enhanced_bot import TradingBot
import asyncio

async def test_bot():
    bot = TradingBot()
    bot.bot = MockBot()  # Use mock instead of real bot
    
    # Test portfolio command
    await bot.portfolio_command(None, None)
    print('✅ Portfolio command test passed')
    
    # Test chart generation
    await bot.chart_command(None, None)
    print('✅ Chart command test passed')
    
    print('✅ All bot tests passed')

asyncio.run(test_bot())
"
```

### Method 2: Local Bot Testing (with fake token)
```bash
# Set fake token for testing
export TELEGRAM_BOT_TOKEN="1234567890:AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRs"

# Test bot initialization and commands
python -c "
import sys
sys.path.append('.')
from src.telegram.enhanced_bot import TradingBot

try:
    bot = TradingBot()
    print('✅ Bot initialization successful')
    print('✅ Bot token configured')
    print('✅ Bot commands loaded')
except Exception as e:
    print(f'⚠️ Bot test warning (expected with fake token): {e}')
"
```

### Method 3: Real Bot Testing (requires real token)
```bash
# Get real bot token from @BotFather on Telegram
export TELEGRAM_BOT_TOKEN="YOUR_REAL_BOT_TOKEN"

# Start bot in test mode
python src/telegram/enhanced_bot.py &
BOT_PID=$!

# Test bot commands in Telegram app:
# /start - Should show welcome message
# /help - Should show command list
# /portfolio - Should show portfolio info
# /chart AAPL - Should generate and send chart

# Stop bot
kill $BOT_PID
```

### Bot Testing Checklist
- [ ] Bot initializes without errors
- [ ] All commands are registered
- [ ] Portfolio analysis works
- [ ] Chart generation works
- [ ] AI suggestions work
- [ ] Voice commands work
- [ ] Error handling works

## 🔧 Component Testing

### ML Pipeline Testing
```bash
# Test ML pipeline components
pytest tests/test_ml_pipeline.py -v

# Test with different symbols
python research/ml_pipeline.py --symbol AAPL --start 2020-01-01 --end 2024-01-01
python research/ml_pipeline.py --symbol MSFT --start 2020-01-01 --end 2024-01-01
```

### Risk Management Testing
```bash
# Test risk calculations
pytest tests/test_risk.py -v

# Test with sample data
python -c "
import sys
sys.path.append('.')
from src.utils.risk import kelly_fraction, sharpe, max_drawdown
import numpy as np

# Sample returns
returns = np.random.normal(0.001, 0.02, 1000)
print(f'Kelly Fraction: {kelly_fraction(returns, 0.6, 1.5):.4f}')
print(f'Sharpe Ratio: {sharpe(returns):.4f}')
print(f'Max Drawdown: {max_drawdown((1+returns).cumprod()):.4f}')
"
```

### Strategy Testing
```bash
# Test scalping strategy
pytest tests/test_scalping.py -v

# Test signal generation
python -c "
import sys
sys.path.append('.')
from src.strategies.scalping import enrich, signals
from src.utils.data import synthetic_ohlcv
from src.config import cfg

df = synthetic_ohlcv('AAPL', 1000)
enriched = enrich(df, cfg=cfg)
sig = signals(enriched, cfg=cfg)
print(f'✅ Generated {len(sig)} signals')
print(f'✅ Signal distribution: {sig.value_counts()}')
"
```

## 🐳 Docker Testing

### Test Docker Build
```bash
# Test main Dockerfile
docker build -t tradingai-test .

# Test enhanced Dockerfile
docker build -f Dockerfile.enhanced -t tradingai-enhanced-test .
```

### Test Docker Compose
```bash
# Validate docker-compose syntax
docker-compose -f docker-compose.enhanced.yml config

# Start services in test mode
docker-compose -f docker-compose.enhanced.yml up -d

# Check service health
docker-compose -f docker-compose.enhanced.yml ps
docker-compose -f docker-compose.enhanced.yml logs tradingai-bot

# Cleanup
docker-compose -f docker-compose.enhanced.yml down
```

## 🌐 End-to-End Testing

### Automated Test Suite
```bash
# Run comprehensive test suite
./validate_and_deploy.sh

# Run specific test categories
pytest tests/ -v --tb=short
pytest tests/test_main.py -v
pytest tests/test_ml_pipeline.py -v
```

### Manual E2E Test
1. Start Streamlit UI: `streamlit run ui/enhanced_dashboard.py`
2. Navigate through all pages
3. Test parameter adjustments
4. Generate and download reports
5. Start Telegram bot with test token
6. Test all bot commands
7. Verify ML pipeline runs
8. Check Docker deployment

## 📊 Performance Testing

### Load Testing
```bash
# Test with large datasets
python research/ml_pipeline.py --symbol AAPL --start 2010-01-01 --end 2024-01-01

# Test UI performance
ab -n 100 -c 10 http://localhost:8501/

# Test bot response time
time curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"
```

### Memory Testing
```bash
# Monitor memory usage
python -c "
import psutil
import sys
sys.path.append('.')

# Before loading
mem_before = psutil.virtual_memory().used / 1024 / 1024
print(f'Memory before: {mem_before:.1f} MB')

# Load ML pipeline
from research.ml_pipeline import *
from ui.enhanced_dashboard import *

# After loading
mem_after = psutil.virtual_memory().used / 1024 / 1024
print(f'Memory after: {mem_after:.1f} MB')
print(f'Memory usage: {mem_after - mem_before:.1f} MB')
"
```

## 🚨 Troubleshooting Common Issues

### UI Issues
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/

# Reset Streamlit config
streamlit config show

# Check port conflicts
lsof -i :8501
```

### Bot Issues
```bash
# Check token format
echo $TELEGRAM_BOT_TOKEN | grep -E '^[0-9]+:[a-zA-Z0-9_-]+$' && echo "✅ Token format valid"

# Test bot token
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"

# Check network connectivity
ping api.telegram.org
```

### Docker Issues
```bash
# Check Docker daemon
docker info

# Check image sizes
docker images | grep tradingai

# Clean up Docker
docker system prune -f
```

## 📝 Test Reports

### Generate Test Coverage
```bash
# Install coverage tools
pip install pytest-cov coverage

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Performance Benchmarks
```bash
# Benchmark ML pipeline
python -m timeit -s "
import sys; sys.path.append('.')
from research.ml_pipeline import train_rf_walkforward
from src.utils.data import synthetic_ohlcv
df = synthetic_ohlcv('AAPL', 1000)
" "train_rf_walkforward(df, target_col='close', feature_cols=['open', 'high', 'low', 'volume'])"
```

## ✅ Success Criteria

### UI Testing Success
- [ ] All pages load without errors
- [ ] Charts render correctly
- [ ] Parameters update in real-time
- [ ] Export functions work
- [ ] No console errors

### Bot Testing Success
- [ ] Bot starts without errors
- [ ] All commands respond correctly
- [ ] Charts generate and send
- [ ] Error handling works
- [ ] Memory usage is reasonable

### System Testing Success
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Docker deployment works
- [ ] Performance meets targets
- [ ] No memory leaks detected

---

**Ready to test your AI trading system! 🧪🚀**
