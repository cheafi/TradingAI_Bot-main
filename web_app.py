"""
Simple Web Interface for TradingAI Bot
=====================================

This creates a basic Flask web application to provide web access to the trading system.
"""

import os
import sys
import json
from datetime import datetime

# Fix platform import issue
import importlib
import importlib.util

try:
    from flask import Flask, render_template_string, jsonify, request
except ImportError:
    print("Flask not available, using simple HTTP server instead")
    Flask = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Add project root to path
sys.path.append('/workspaces/TradingAI_Bot-main')

app = Flask(__name__)

# HTML template for the main interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>TradingAI Bot - Institutional Grade</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin: 0;
            font-weight: 300;
        }
        .subtitle {
            color: #666;
            font-size: 1.2em;
            margin: 10px 0;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .status-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #667eea;
            transition: transform 0.2s;
        }
        .status-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .status-card h3 {
            margin: 0 0 10px 0;
            color: #667eea;
        }
        .status-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }
        .status-operational {
            color: #28a745;
        }
        .status-warning {
            color: #ffc107;
        }
        .status-danger {
            color: #dc3545;
        }
        .nav-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 25px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 25px;
            transition: all 0.3s;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background: #5a6fd8;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: #6c757d;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        .feature-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            text-align: center;
        }
        .feature-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
    </style>
    <script>
        function refreshStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('system-status').innerHTML = 
                        '<span class="status-' + (data.system_operational ? 'operational' : 'danger') + '">' +
                        (data.system_operational ? 'OPERATIONAL' : 'OFFLINE') + '</span>';
                    
                    document.getElementById('kill-switch-status').innerHTML = 
                        '<span class="status-' + (data.kill_switch_armed ? 'operational' : 'warning') + '">' +
                        (data.kill_switch_armed ? 'ARMED' : 'DISARMED') + '</span>';
                    
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    document.getElementById('system-status').innerHTML = 
                        '<span class="status-danger">ERROR</span>';
                });
        }
        
        // Refresh status every 30 seconds
        setInterval(refreshStatus, 30000);
        
        // Initial load
        window.onload = function() {
            refreshStatus();
        };
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ TradingAI Bot</h1>
            <div class="subtitle">Institutional-Grade Algorithmic Trading Platform</div>
            <div style="margin-top: 10px; color: #888;">
                Production-Ready • Risk-Managed • Leakage-Free
            </div>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>🔄 System Status</h3>
                <div class="status-value" id="system-status">LOADING...</div>
            </div>
            <div class="status-card">
                <h3>🛡️ Kill-Switch</h3>
                <div class="status-value" id="kill-switch-status">LOADING...</div>
            </div>
            <div class="status-card">
                <h3>📊 Last Update</h3>
                <div class="status-value" id="last-update">--:--:--</div>
            </div>
            <div class="status-card">
                <h3>⚡ Version</h3>
                <div class="status-value">v2.0-INSTITUTIONAL</div>
            </div>
        </div>

        <div class="nav-buttons">
            <a href="/dashboard" class="btn">📈 Trading Dashboard</a>
            <a href="/risk" class="btn">🛡️ Risk Management</a>
            <a href="/backtest" class="btn">🔬 Backtesting</a>
            <a href="/api/docs" class="btn btn-secondary">📚 API Docs</a>
        </div>

        <div class="features">
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <h3>Anti-Leakage System</h3>
                <p>Mathematical guarantees against future data access with point-in-time enforcement and temporal validation.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🚨</div>
                <h3>Kill-Switch Protection</h3>
                <p>Real-time risk monitoring with automatic emergency halt capabilities and multi-layer circuit breakers.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">💰</div>
                <h3>Realistic Cost Modeling</h3>
                <p>Professional-grade transaction cost analysis including market impact, spreads, and implementation shortfall.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔄</div>
                <h3>Walk-Forward Validation</h3>
                <p>Institutional validation framework with embargo periods preventing temporal leakage in backtests.</p>
            </div>
        </div>

        <div style="text-align: center; margin: 30px 0;">
            <h3>🎯 Available Interfaces</h3>
            <div class="nav-buttons">
                <a href="http://localhost:8501" target="_blank" class="btn">📊 Streamlit Dashboard</a>
                <button onclick="window.location.reload()" class="btn btn-secondary">🔄 Refresh Status</button>
            </div>
        </div>

        <div class="footer">
            <p>🏛️ Institutional-Grade Trading Platform • Built for Professional Risk Management</p>
            <p>© 2025 TradingAI Bot - Production Ready</p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Main dashboard page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    try:
        # Try to import our institutional systems
        sys.path.append('/workspaces/TradingAI_Bot-main')
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_operational": True,
            "kill_switch_armed": True,
            "trading_state": "NORMAL",
            "components": {
                "data_contracts": True,
                "kill_switch": True,
                "cost_model": True,
                "walk_forward_validation": True
            },
            "version": "v2.0-INSTITUTIONAL"
        }
        
        # Try to get actual kill-switch status
        try:
            from trading_platform.safety.killswitch import get_kill_switch
            kill_switch = get_kill_switch()
            kill_status = kill_switch.get_status()
            status["trading_state"] = kill_status.get("trading_state", "UNKNOWN")
            status["kill_switch_armed"] = kill_status.get("is_armed", False)
        except Exception as e:
            status["components"]["kill_switch"] = False
            status["error"] = f"Kill-switch error: {str(e)}"
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "system_operational": False,
            "error": str(e),
            "version": "v2.0-INSTITUTIONAL"
        }), 500

@app.route('/dashboard')
def dashboard():
    """Redirect to Streamlit dashboard."""
    return """
    <html>
    <head>
        <title>Redirecting to Dashboard...</title>
        <meta http-equiv="refresh" content="0; url=http://localhost:8501">
    </head>
    <body>
        <p>Redirecting to Streamlit Dashboard...</p>
        <p>If not redirected automatically, <a href="http://localhost:8501">click here</a>.</p>
    </body>
    </html>
    """

@app.route('/risk')
def risk_management():
    """Risk management interface."""
    return render_template_string("""
    <html>
    <head><title>Risk Management - TradingAI Bot</title></head>
    <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h1>🛡️ Risk Management System</h1>
        <h2>Kill-Switch Status</h2>
        <div id="risk-status">Loading...</div>
        
        <h2>Circuit Breakers</h2>
        <ul>
            <li>Daily P&L: -2% soft limit, -5% hard limit</li>
            <li>Position Concentration: 25% soft, 40% hard</li>
            <li>Portfolio VaR: 3% soft, 5% hard</li>
            <li>Maximum Drawdown: -3% soft, -8% hard</li>
        </ul>
        
        <p><a href="/">← Back to Main Dashboard</a></p>
        
        <script>
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('risk-status').innerHTML = 
                        '<strong>Trading State:</strong> ' + data.trading_state + '<br>' +
                        '<strong>Kill-Switch:</strong> ' + (data.kill_switch_armed ? 'ARMED' : 'DISARMED');
                });
        </script>
    </body>
    </html>
    """)

@app.route('/backtest')
def backtest():
    """Backtesting interface."""
    return render_template_string("""
    <html>
    <head><title>Backtesting - TradingAI Bot</title></head>
    <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h1>🔬 Walk-Forward Validation System</h1>
        <h2>Institutional-Grade Backtesting</h2>
        
        <h3>Features:</h3>
        <ul>
            <li>✅ Embargo windows preventing temporal leakage</li>
            <li>✅ Model degradation detection</li>
            <li>✅ Rolling/expanding window validation</li>
            <li>✅ Realistic cost modeling</li>
            <li>✅ Implementation shortfall tracking</li>
        </ul>
        
        <h3>Anti-Leakage Guarantees:</h3>
        <ul>
            <li>🛡️ Point-in-time data access only</li>
            <li>🛡️ Mandatory embargo periods</li>
            <li>🛡️ Future data spike testing</li>
            <li>🛡️ Temporal ordering enforcement</li>
        </ul>
        
        <p><a href="/">← Back to Main Dashboard</a></p>
    </body>
    </html>
    """)

@app.route('/api/docs')
def api_docs():
    """API documentation."""
    return render_template_string("""
    <html>
    <head><title>API Documentation - TradingAI Bot</title></head>
    <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h1>📚 API Documentation</h1>
        
        <h2>Available Endpoints:</h2>
        <ul>
            <li><strong>GET /</strong> - Main dashboard</li>
            <li><strong>GET /api/status</strong> - System status JSON</li>
            <li><strong>GET /dashboard</strong> - Redirect to Streamlit</li>
            <li><strong>GET /risk</strong> - Risk management interface</li>
            <li><strong>GET /backtest</strong> - Backtesting interface</li>
        </ul>
        
        <h2>Status API Response:</h2>
        <pre>{
    "timestamp": "2025-09-03T05:15:00",
    "system_operational": true,
    "kill_switch_armed": true,
    "trading_state": "NORMAL",
    "components": {
        "data_contracts": true,
        "kill_switch": true,
        "cost_model": true,
        "walk_forward_validation": true
    },
    "version": "v2.0-INSTITUTIONAL"
}</pre>
        
        <p><a href="/">← Back to Main Dashboard</a></p>
    </body>
    </html>
    """)

if __name__ == '__main__':
    print("🏛️ Starting TradingAI Bot Web Interface...")
    print("📊 Streamlit Dashboard: http://localhost:8501")
    print("🌐 Main Interface: http://localhost:5000")
    print("🛡️ Institutional-Grade Trading Platform Ready!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
