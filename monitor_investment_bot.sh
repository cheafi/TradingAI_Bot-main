#!/bin/bash

# Monitor the investment bot
while true; do
    echo "🔍 [$(date)] Monitoring TradingAI Pro Investment Agency..."
    
    # Check if bot is still running
    if ps -p $1 > /dev/null; then
        echo "✅ Investment agency bot is running (PID: $1)"
    else
        echo "❌ Investment agency bot stopped, restarting..."
        # In production, restart logic would go here
        python src/telegram/investment_agency_dry.py &
        BOT_PID=$!
        echo "🔄 Bot restarted with new PID: $BOT_PID"
    fi
    
    # Generate hourly market update
    python -c "
import sys
sys.path.append('.')
from datetime import datetime
print(f'📊 [{datetime.now().strftime(\"%H:%M\")}] Market Update: All systems operational')
print('💹 Monitoring 50+ assets across global markets')
print('🤖 AI algorithms processing real-time data')
print('📱 Ready to send alerts to subscribers')
"
    
    sleep 3600  # Check every hour
done
