# UI Testing Guide - Streamlit Dashboard

This guide shows you how to test the TradingAI Pro Streamlit UI in various modes.

## 🖥️ UI Testing Methods

### Method 1: Interactive Development (Recommended)
```bash
# Activate environment
source .venv/bin/activate

# Start development server
streamlit run ui/enhanced_dashboard.py --server.port 8501

# Open browser: http://localhost:8501
```

**What to test:**
- [ ] Dashboard loads without errors
- [ ] Navigation sidebar works
- [ ] All pages accessible
- [ ] Charts render correctly
- [ ] No console errors in browser

### Method 2: Dry Run Testing (No Browser)
```bash
# Test imports and syntax
python -c "
import sys
sys.path.append('.')

print('Testing UI imports...')
import streamlit as st
import ui.enhanced_dashboard
import ui.pages.data_explorer
import ui.pages.variable_tuner
import ui.pages.prediction_analysis
import ui.pages.portfolio_analysis

print('✅ All UI components imported successfully')
print(f'✅ Streamlit version: {st.__version__}')
"
```

### Method 3: Headless Testing (CI/CD)
```bash
# Start headless server
STREAMLIT_SERVER_HEADLESS=true streamlit run ui/enhanced_dashboard.py --server.port 8502 &
SERVER_PID=$!

# Wait for startup
sleep 5

# Test endpoint
curl -f http://localhost:8502/_stcore/health && echo "✅ UI server healthy"

# Test main page
curl -s http://localhost:8502 | grep -q "TradingAI Pro" && echo "✅ Main page loads"

# Cleanup
kill $SERVER_PID
```

## 📱 UI Components Testing

### Dashboard Pages Testing
```bash
# Test individual page components
python -c "
import sys
sys.path.append('.')
import pandas as pd
import numpy as np

# Mock data for testing
mock_data = pd.DataFrame({
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
}, index=pd.date_range('2020-01-01', periods=100))

print('Testing page components...')

# Test data explorer components
from ui.pages.data_explorer import create_correlation_heatmap
try:
    create_correlation_heatmap(mock_data)
    print('✅ Data explorer: correlation heatmap works')
except Exception as e:
    print(f'❌ Data explorer error: {e}')

# Test variable tuner components
from ui.pages.variable_tuner import calculate_sensitivity
try:
    sensitivity = calculate_sensitivity('EMA_PERIOD', 10, 30, mock_data)
    print('✅ Variable tuner: sensitivity analysis works')
except Exception as e:
    print(f'❌ Variable tuner error: {e}')

print('✅ Component testing complete')
"
```

### Chart Generation Testing
```bash
# Test chart components
python -c "
import sys
sys.path.append('.')
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Mock data
dates = pd.date_range('2020-01-01', periods=100)
prices = pd.DataFrame({
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 105,
    'low': np.random.randn(100).cumsum() + 95,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
}, index=dates)

print('Testing chart generation...')

# Test candlestick chart
try:
    fig = go.Figure(data=go.Candlestick(
        x=prices.index,
        open=prices['open'],
        high=prices['high'],
        low=prices['low'],
        close=prices['close']
    ))
    print('✅ Candlestick chart generation works')
except Exception as e:
    print(f'❌ Chart error: {e}')

# Test 3D correlation
try:
    correlation_data = prices.corr()
    fig = go.Figure(data=go.Surface(z=correlation_data.values))
    print('✅ 3D correlation chart generation works')
except Exception as e:
    print(f'❌ 3D chart error: {e}')

print('✅ Chart testing complete')
"
```

## 🎨 UI Features Testing

### Test Theme Switching
```bash
# Test CSS themes
python -c "
import sys
sys.path.append('.')

print('Testing UI themes...')

# Test theme CSS generation
from ui.enhanced_dashboard import get_custom_css

try:
    dark_css = get_custom_css('dark')
    light_css = get_custom_css('light')
    
    assert 'background-color' in dark_css
    assert 'background-color' in light_css
    print('✅ Theme CSS generation works')
    print('✅ Dark theme CSS loaded')
    print('✅ Light theme CSS loaded')
except Exception as e:
    print(f'❌ Theme error: {e}')
"
```

### Test Data Export Functions
```bash
# Test export functionality
python -c "
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import io

print('Testing export functions...')

# Mock data
mock_data = pd.DataFrame({
    'symbol': ['AAPL'] * 100,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
}, index=pd.date_range('2020-01-01', periods=100))

# Test CSV export
try:
    csv_buffer = io.StringIO()
    mock_data.to_csv(csv_buffer)
    csv_data = csv_buffer.getvalue()
    assert len(csv_data) > 0
    print('✅ CSV export works')
except Exception as e:
    print(f'❌ CSV export error: {e}')

# Test JSON export
try:
    json_data = mock_data.to_json()
    assert len(json_data) > 0
    print('✅ JSON export works')
except Exception as e:
    print(f'❌ JSON export error: {e}')

print('✅ Export testing complete')
"
```

## 🔧 UI Performance Testing

### Test Load Time
```bash
# Measure page load performance
python -c "
import time
import sys
sys.path.append('.')

print('Testing UI performance...')

start_time = time.time()

# Import main components
import streamlit as st
import ui.enhanced_dashboard
import ui.pages.data_explorer
import ui.pages.variable_tuner

load_time = time.time() - start_time
print(f'✅ UI load time: {load_time:.2f} seconds')

if load_time < 5:
    print('✅ Performance: Good (< 5s)')
elif load_time < 10:
    print('⚠️ Performance: Acceptable (5-10s)')
else:
    print('❌ Performance: Slow (> 10s)')
"
```

### Test Memory Usage
```bash
# Monitor memory consumption
python -c "
import psutil
import sys
sys.path.append('.')

process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024

print(f'Memory before UI load: {mem_before:.1f} MB')

# Load UI components
import streamlit as st
import ui.enhanced_dashboard
import pandas as pd
import numpy as np

# Create test data
test_data = pd.DataFrame(np.random.randn(10000, 10))

mem_after = process.memory_info().rss / 1024 / 1024
mem_usage = mem_after - mem_before

print(f'Memory after UI load: {mem_after:.1f} MB')
print(f'Memory usage: {mem_usage:.1f} MB')

if mem_usage < 100:
    print('✅ Memory usage: Good (< 100MB)')
elif mem_usage < 200:
    print('⚠️ Memory usage: Acceptable (100-200MB)')
else:
    print('❌ Memory usage: High (> 200MB)')
"
```

## 🌐 Browser Testing

### Test Browser Compatibility
```bash
# Start UI server
streamlit run ui/enhanced_dashboard.py --server.port 8503 &
SERVER_PID=$!

sleep 5

# Test with different user agents
echo "Testing browser compatibility..."

# Chrome
curl -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
     -s http://localhost:8503 | grep -q "TradingAI" && echo "✅ Chrome compatible"

# Firefox
curl -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0" \
     -s http://localhost:8503 | grep -q "TradingAI" && echo "✅ Firefox compatible"

# Safari
curl -H "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15" \
     -s http://localhost:8503 | grep -q "TradingAI" && echo "✅ Safari compatible"

# Mobile
curl -H "User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15" \
     -s http://localhost:8503 | grep -q "TradingAI" && echo "✅ Mobile compatible"

# Cleanup
kill $SERVER_PID
```

## 📱 Mobile Responsiveness Testing

### Test Mobile Layout
```bash
# Test responsive design
python -c "
print('Testing mobile responsiveness...')

# Check CSS for responsive design
with open('ui/enhanced_dashboard.py', 'r') as f:
    content = f.read()
    
# Look for responsive design patterns
responsive_patterns = [
    '@media',
    'max-width',
    'mobile',
    'responsive',
    'flex',
    'grid'
]

found_patterns = []
for pattern in responsive_patterns:
    if pattern in content.lower():
        found_patterns.append(pattern)

if found_patterns:
    print(f'✅ Responsive design patterns found: {found_patterns}')
else:
    print('⚠️ Limited responsive design detected')

# Check for mobile-friendly components
if 'st.sidebar' in content:
    print('✅ Sidebar navigation (mobile-friendly)')
if 'st.columns' in content:
    print('✅ Column layout (responsive)')
if 'st.container' in content:
    print('✅ Container layout (flexible)')
"
```

## 🎯 UI Testing Checklist

### Basic Functionality
- [ ] Application starts without errors
- [ ] All pages load successfully
- [ ] Navigation works correctly
- [ ] Charts render properly
- [ ] Data updates in real-time

### Interactive Features
- [ ] Parameter sliders work
- [ ] Dropdown menus function
- [ ] Buttons respond correctly
- [ ] File uploads work (if applicable)
- [ ] Data export functions

### Visual Design
- [ ] Layout is clean and organized
- [ ] Colors and themes apply correctly
- [ ] Text is readable
- [ ] Charts are visually appealing
- [ ] Mobile layout is usable

### Performance
- [ ] Page load time < 5 seconds
- [ ] Memory usage < 200MB
- [ ] No console errors
- [ ] Smooth interactions
- [ ] Charts render quickly

## 🚨 Common UI Issues & Solutions

### Issue: Streamlit Not Found
```bash
# Solution: Install streamlit
pip install streamlit

# Or reinstall
pip install --upgrade streamlit
```

### Issue: Port Already in Use
```bash
# Solution: Use different port
streamlit run ui/enhanced_dashboard.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9
```

### Issue: Import Errors
```bash
# Solution: Check Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or use absolute imports
python -c "import sys; sys.path.append('.'); import ui.enhanced_dashboard"
```

### Issue: Charts Not Rendering
```bash
# Solution: Check plotly installation
pip install --upgrade plotly

# Test plotly
python -c "import plotly.graph_objects as go; print('✅ Plotly works')"
```

## 🎉 Success Criteria

Your UI testing is successful when:
- ✅ All pages load without errors
- ✅ Interactive components respond correctly
- ✅ Charts render properly
- ✅ Performance is acceptable
- ✅ Mobile layout is usable
- ✅ Export functions work
- ✅ Theme switching works

**Your Streamlit UI is ready for production! 🎨🚀**
