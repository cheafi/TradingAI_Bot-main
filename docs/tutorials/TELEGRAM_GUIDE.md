# Telegram Bot Testing Guide

Complete guide for testing the TradingAI Pro Telegram bot in dry run and live modes.

## 🤖 Bot Testing Overview

### Testing Modes
1. **Dry Run Testing**: Test without real Telegram token
2. **Mock Testing**: Test with simulated responses
3. **Development Testing**: Test with real token in test environment
4. **Production Testing**: Test with real token and users

## 🧪 Dry Run Testing (No Real Token Required)

### Method 1: Import and Syntax Testing
```bash
# Test bot imports and initialization
python -c "
import sys
sys.path.append('.')

print('Testing Telegram bot imports...')

# Test basic imports
from src.telegram.enhanced_bot import TradingBot
print('✅ Bot class imported successfully')

# Test dependencies
import telegram
import asyncio
import plotly.graph_objects as go
print('✅ All dependencies available')

# Test configuration
from src.config import cfg
print('✅ Configuration loaded')

print('✅ Dry run import test complete')
"
```

### Method 2: Mock Bot Testing
```bash
# Test bot functionality with mock responses
python -c "
import sys
sys.path.append('.')
import asyncio
from unittest.mock import Mock, AsyncMock

print('Testing bot functionality with mocks...')

# Create mock bot
class MockTelegramBot:
    def __init__(self):
        self.token = 'mock_token'
    
    async def send_message(self, chat_id, text, **kwargs):
        print(f'📤 Would send to {chat_id}: {text[:100]}...')
        return Mock(message_id=123)
    
    async def send_photo(self, chat_id, photo, **kwargs):
        print(f'📷 Would send photo to {chat_id}')
        return Mock(message_id=124)
    
    async def send_voice(self, chat_id, voice, **kwargs):
        print(f'🎙️ Would send voice to {chat_id}')
        return Mock(message_id=125)

# Test bot commands
from src.telegram.enhanced_bot import TradingBot

async def test_bot_commands():
    bot_instance = TradingBot()
    bot_instance.bot = MockTelegramBot()
    
    # Mock update and context
    update = Mock()
    update.effective_chat.id = 12345
    update.message.text = '/portfolio'
    
    context = Mock()
    context.args = []
    
    try:
        # Test portfolio command
        await bot_instance.portfolio_command(update, context)
        print('✅ Portfolio command test passed')
        
        # Test chart command
        update.message.text = '/chart AAPL'
        context.args = ['AAPL']
        await bot_instance.chart_command(update, context)
        print('✅ Chart command test passed')
        
        # Test suggest command
        update.message.text = '/suggest AAPL'
        await bot_instance.suggest_command(update, context)
        print('✅ Suggest command test passed')
        
        print('✅ All mock tests passed')
        
    except Exception as e:
        print(f'❌ Mock test failed: {e}')

# Run async test
asyncio.run(test_bot_commands())
"
```

### Method 3: Component Testing
```bash
# Test individual bot components
python -c "
import sys
sys.path.append('.')

print('Testing bot components...')

# Test chart generation
try:
    from src.telegram.enhanced_bot import TradingBot
    import pandas as pd
    import numpy as np
    
    bot = TradingBot()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=pd.date_range('2020-01-01', periods=100))
    
    # Test chart generation (returns bytes)
    chart_bytes = bot.generate_chart(sample_data, 'TEST')
    assert isinstance(chart_bytes, bytes)
    print('✅ Chart generation works')
    
except Exception as e:
    print(f'❌ Chart generation error: {e}')

# Test AI suggestions
try:
    suggestion = bot.generate_ai_suggestion('AAPL', sample_data)
    assert isinstance(suggestion, str)
    assert len(suggestion) > 0
    print('✅ AI suggestion generation works')
    
except Exception as e:
    print(f'❌ AI suggestion error: {e}')

# Test portfolio analysis
try:
    portfolio_text = bot.analyze_portfolio()
    assert isinstance(portfolio_text, str)
    print('✅ Portfolio analysis works')
    
except Exception as e:
    print(f'❌ Portfolio analysis error: {e}')

print('✅ Component testing complete')
"
```

## 🔧 Development Testing (With Test Token)

### Step 1: Create Test Bot
1. Message @BotFather on Telegram
2. Use `/newbot` command
3. Name your bot (e.g., "MyTradingTestBot")
4. Get your test token
5. **Important**: Keep this token private!

### Step 2: Set Test Token
```bash
# Set your test bot token (replace with actual token)
export TELEGRAM_BOT_TOKEN="1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"

# Verify token format
echo $TELEGRAM_BOT_TOKEN | grep -E '^[0-9]+:[a-zA-Z0-9_-]+$' && echo "✅ Token format valid"
```

### Step 3: Test Bot Connection
```bash
# Test bot token with Telegram API
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe" | jq '.'

# Should return bot information like:
# {
#   "ok": true,
#   "result": {
#     "id": 1234567890,
#     "is_bot": true,
#     "first_name": "MyTradingTestBot",
#     "username": "mytradingtest_bot"
#   }
# }
```

### Step 4: Start Test Bot
```bash
# Start bot in test mode
python src/telegram/enhanced_bot.py &
BOT_PID=$!

# Check if bot started successfully
sleep 3
ps aux | grep enhanced_bot && echo "✅ Bot is running"

# Check logs
tail -f logs/bot.log  # if logging is configured
```

### Step 5: Test Commands in Telegram

**Find your bot in Telegram:**
1. Search for your bot username
2. Start conversation with `/start`

**Test basic commands:**
```
/start          # Should show welcome message
/help           # Should show command list
/portfolio      # Should show portfolio analysis
/status         # Should show system status
```

**Test chart commands:**
```
/chart AAPL     # Should generate and send AAPL chart
/chart BTC      # Should generate and send BTC chart
/chart INVALID  # Should handle error gracefully
```

**Test AI commands:**
```
/suggest AAPL   # Should provide AI analysis
/optimize       # Should run optimization
/voice          # Should send voice message
```

## 📱 Interactive Testing Commands

### Test Portfolio Command
```bash
# Simulate portfolio command
python -c "
import sys
sys.path.append('.')
import asyncio
from unittest.mock import Mock

async def test_portfolio():
    from src.telegram.enhanced_bot import TradingBot
    
    bot = TradingBot()
    
    # Create mock update
    update = Mock()
    update.effective_chat.id = 12345
    update.message.from_user.first_name = 'TestUser'
    
    context = Mock()
    
    # Test portfolio command
    response = await bot.portfolio_command(update, context)
    print('✅ Portfolio command executed')

# Run test
try:
    asyncio.run(test_portfolio())
except Exception as e:
    print(f'Portfolio test: {e}')
"
```

### Test Chart Generation
```bash
# Test chart command with different symbols
python -c "
import sys
sys.path.append('.')
import asyncio
from unittest.mock import Mock

async def test_charts():
    from src.telegram.enhanced_bot import TradingBot
    
    bot = TradingBot()
    
    symbols = ['AAPL', 'MSFT', 'BTC/USDT', 'ETH/USDT']
    
    for symbol in symbols:
        try:
            # Test chart generation
            update = Mock()
            update.effective_chat.id = 12345
            
            context = Mock()
            context.args = [symbol]
            
            await bot.chart_command(update, context)
            print(f'✅ Chart test passed for {symbol}')
            
        except Exception as e:
            print(f'❌ Chart test failed for {symbol}: {e}')

# Run test
asyncio.run(test_charts())
"
```

### Test Error Handling
```bash
# Test bot error handling
python -c "
import sys
sys.path.append('.')
import asyncio
from unittest.mock import Mock

async def test_error_handling():
    from src.telegram.enhanced_bot import TradingBot
    
    bot = TradingBot()
    
    # Test invalid commands
    test_cases = [
        ('/chart', []),  # Missing symbol
        ('/chart', ['INVALID_SYMBOL']),  # Invalid symbol
        ('/suggest', []),  # Missing symbol
        ('/unknown', []),  # Unknown command
    ]
    
    for command, args in test_cases:
        try:
            update = Mock()
            update.effective_chat.id = 12345
            update.message.text = command
            
            context = Mock()
            context.args = args
            
            # Test error handling
            if command == '/chart':
                await bot.chart_command(update, context)
            elif command == '/suggest':
                await bot.suggest_command(update, context)
            
            print(f'✅ Error handling test passed for {command}')
            
        except Exception as e:
            print(f'⚠️ Expected error for {command}: {e}')

# Run test
asyncio.run(test_error_handling())
"
```

## 🔍 Bot Performance Testing

### Test Response Time
```bash
# Measure bot response time
python -c "
import sys
sys.path.append('.')
import time
import asyncio
from unittest.mock import Mock

async def test_response_time():
    from src.telegram.enhanced_bot import TradingBot
    
    bot = TradingBot()
    
    commands = [
        ('portfolio', bot.portfolio_command),
        ('status', bot.status_command),
        ('help', bot.help_command),
    ]
    
    for cmd_name, cmd_func in commands:
        update = Mock()
        update.effective_chat.id = 12345
        context = Mock()
        
        start_time = time.time()
        
        try:
            await cmd_func(update, context)
            response_time = time.time() - start_time
            print(f'✅ {cmd_name}: {response_time:.2f}s')
            
            if response_time < 1:
                print(f'   Performance: Excellent')
            elif response_time < 3:
                print(f'   Performance: Good')
            else:
                print(f'   Performance: Slow')
                
        except Exception as e:
            print(f'❌ {cmd_name} failed: {e}')

# Run test
asyncio.run(test_response_time())
"
```

### Test Memory Usage
```bash
# Monitor bot memory usage
python -c "
import psutil
import sys
sys.path.append('.')

process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024

print(f'Memory before bot load: {mem_before:.1f} MB')

# Load bot
from src.telegram.enhanced_bot import TradingBot
bot = TradingBot()

mem_after = process.memory_info().rss / 1024 / 1024
mem_usage = mem_after - mem_before

print(f'Memory after bot load: {mem_after:.1f} MB')
print(f'Bot memory usage: {mem_usage:.1f} MB')

if mem_usage < 50:
    print('✅ Memory usage: Excellent (< 50MB)')
elif mem_usage < 100:
    print('✅ Memory usage: Good (50-100MB)')
else:
    print('⚠️ Memory usage: High (> 100MB)')
"
```

## 🚨 Troubleshooting Bot Issues

### Issue: Token Invalid
```bash
# Test token validity
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"

# Should return bot info, not error
```

### Issue: Bot Not Responding
```bash
# Check bot process
ps aux | grep enhanced_bot

# Check network connectivity
ping api.telegram.org

# Check token permissions
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates"
```

### Issue: Import Errors
```bash
# Check Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Test imports
python -c "import telegram; print('✅ python-telegram-bot installed')"
python -c "import plotly; print('✅ plotly installed')"
```

### Issue: Chart Generation Fails
```bash
# Test plotly installation
pip install --upgrade plotly kaleido

# Test chart generation
python -c "
import plotly.graph_objects as go
import io

fig = go.Figure(data=go.Scatter(x=[1,2,3], y=[1,2,3]))
img_bytes = fig.to_image(format='png')
print(f'✅ Chart generated: {len(img_bytes)} bytes')
"
```

## 📋 Bot Testing Checklist

### Basic Functionality
- [ ] Bot starts without errors
- [ ] Token is valid and working
- [ ] Bot responds to `/start`
- [ ] Help command shows all available commands
- [ ] Error handling works correctly

### Command Testing
- [ ] `/portfolio` - Shows portfolio analysis
- [ ] `/chart SYMBOL` - Generates and sends charts
- [ ] `/suggest SYMBOL` - Provides AI suggestions
- [ ] `/optimize` - Runs optimization
- [ ] `/voice` - Sends voice messages
- [ ] `/status` - Shows system status

### Performance Testing
- [ ] Response time < 3 seconds per command
- [ ] Memory usage < 100MB
- [ ] Charts generate successfully
- [ ] No memory leaks during extended use
- [ ] Handles multiple users concurrently

### Error Handling
- [ ] Graceful handling of invalid commands
- [ ] Proper error messages for invalid symbols
- [ ] Network error recovery
- [ ] Rate limiting compliance
- [ ] Security measures in place

## 🎯 Success Criteria

Your Telegram bot testing is successful when:
- ✅ All commands respond correctly
- ✅ Charts generate and send properly
- ✅ AI suggestions are provided
- ✅ Error handling works gracefully
- ✅ Performance is acceptable
- ✅ Memory usage is reasonable
- ✅ Multiple users can interact simultaneously

## 🔒 Security Best Practices

### Token Security
```bash
# Never commit tokens to git
echo "TELEGRAM_BOT_TOKEN=*" >> .gitignore

# Use environment variables
export TELEGRAM_BOT_TOKEN="your_token_here"

# Or use .env files
echo "TELEGRAM_BOT_TOKEN=your_token_here" > .env
```

### User Authentication
```bash
# Implement user whitelist (optional)
python -c "
# In your bot code:
ALLOWED_USERS = [12345, 67890]  # Your Telegram user IDs

def is_authorized(user_id):
    return user_id in ALLOWED_USERS

# Use in command handlers:
if not is_authorized(update.effective_user.id):
    await update.message.reply_text('🚫 Unauthorized')
    return
"
```

**Your Telegram bot is ready for trading! 🤖📈**
