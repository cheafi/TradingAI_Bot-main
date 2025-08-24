"""
Daily Market Reports Generator
Generates automated daily reports for the investment agency
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.utils.data import fetch_ohlcv_ccxt, synthetic_ohlcv


class DailyReportsGenerator:
    """Generate automated daily market reports"""
    
    def __init__(self):
        self.reports_dir = "data/daily_reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        
    async def generate_all_reports(self):
        """Generate all daily reports"""
        print("📊 Generating Daily Market Reports...")
        
        # Generate market snapshot
        await self.generate_market_snapshot()
        
        # Generate sector analysis
        await self.generate_sector_analysis()
        
        # Generate opportunities report
        await self.generate_opportunities_report()
        
        # Generate risk assessment
        await self.generate_risk_assessment()
        
        print("✅ All daily reports generated successfully")
    
    async def generate_market_snapshot(self):
        """Generate market snapshot data"""
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT',
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
            'SPY', 'QQQ', 'IWM', 'GLD', 'TLT'
        ]
        
        market_data = {}
        
        for symbol in symbols:
            try:
                # Use synthetic data for demonstration
                df = synthetic_ohlcv(symbol, limit=100)
                
                current_price = float(df['close'].iloc[-1])
                prev_close = float(df['close'].iloc[-2])
                change_24h = ((current_price / prev_close) - 1) * 100
                
                # Calculate additional metrics
                high_52w = float(df['high'].tail(252).max())
                low_52w = float(df['low'].tail(252).min())
                volume_avg = float(df['volume'].tail(20).mean())
                
                market_data[symbol] = {
                    'current_price': current_price,
                    'change_24h': change_24h,
                    'volume': float(df['volume'].iloc[-1]),
                    'volume_avg_20d': volume_avg,
                    'high_52w': high_52w,
                    'low_52w': low_52w,
                    'from_52w_high': ((current_price / high_52w) - 1) * 100,
                    'from_52w_low': ((current_price / low_52w) - 1) * 100,
                    'last_update': datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        # Save market snapshot
        with open(f"{self.reports_dir}/market_snapshot.json", 'w') as f:
            json.dump(market_data, f, indent=2)
        
        print(f"📈 Market snapshot saved: {len(market_data)} symbols")
    
    async def generate_sector_analysis(self):
        """Generate sector performance analysis"""
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'Cryptocurrency': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT'],
            'ETFs': ['SPY', 'QQQ', 'IWM'],
            'Commodities': ['GLD']
        }
        
        sector_analysis = {}
        
        for sector_name, symbols in sectors.items():
            sector_performance = []
            
            for symbol in symbols:
                try:
                    df = synthetic_ohlcv(symbol, limit=50)
                    
                    # Calculate performance metrics
                    current = float(df['close'].iloc[-1])
                    week_ago = float(df['close'].iloc[-7])
                    month_ago = float(df['close'].iloc[-30])
                    
                    week_return = ((current / week_ago) - 1) * 100
                    month_return = ((current / month_ago) - 1) * 100
                    
                    sector_performance.append({
                        'symbol': symbol,
                        'week_return': week_return,
                        'month_return': month_return
                    })
                    
                except Exception as e:
                    print(f"Error in sector analysis for {symbol}: {e}")
                    continue
            
            if sector_performance:
                avg_week = np.mean([p['week_return'] for p in sector_performance])
                avg_month = np.mean([p['month_return'] for p in sector_performance])
                
                sector_analysis[sector_name] = {
                    'avg_week_return': avg_week,
                    'avg_month_return': avg_month,
                    'performance': sector_performance,
                    'trend': 'Bullish' if avg_week > 2 else 'Bearish' if avg_week < -2 else 'Neutral'
                }
        
        # Save sector analysis
        with open(f"{self.reports_dir}/sector_analysis.json", 'w') as f:
            json.dump(sector_analysis, f, indent=2)
        
        print(f"📊 Sector analysis saved: {len(sector_analysis)} sectors")
    
    async def generate_opportunities_report(self):
        """Generate investment opportunities report"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'BTC/USDT', 'ETH/USDT']
        
        opportunities = []
        
        for symbol in symbols:
            try:
                df = synthetic_ohlcv(symbol, limit=100)
                
                # Calculate technical indicators
                current_price = float(df['close'].iloc[-1])
                sma_20 = float(df['close'].tail(20).mean())
                sma_50 = float(df['close'].tail(50).mean())
                
                # Calculate RSI (simplified)
                deltas = df['close'].diff()
                gain = deltas.where(deltas > 0, 0).rolling(window=14).mean()
                loss = -deltas.where(deltas < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = float(rsi.iloc[-1])
                
                # Calculate volatility
                returns = df['close'].pct_change()
                volatility = float(returns.std() * np.sqrt(252) * 100)
                
                # Determine opportunity score
                score = 0
                reasons = []
                
                if current_price < sma_20 * 0.95:  # Below 20-day SMA
                    score += 2
                    reasons.append("Below key moving average")
                
                if current_rsi < 35:  # Oversold
                    score += 3
                    reasons.append("Oversold conditions")
                
                if sma_20 > sma_50:  # Uptrend
                    score += 1
                    reasons.append("Uptrend intact")
                
                if volatility > 25:  # High volatility
                    score += 1
                    reasons.append("High volatility opportunity")
                
                if score >= 3:  # Minimum score for opportunity
                    opportunities.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'score': score,
                        'rsi': current_rsi,
                        'volatility': volatility,
                        'reasons': reasons,
                        'recommendation': 'BUY' if score >= 5 else 'WATCH',
                        'target_price': current_price * 1.15,
                        'stop_loss': current_price * 0.92
                    })
                    
            except Exception as e:
                print(f"Error in opportunities analysis for {symbol}: {e}")
                continue
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Save opportunities report
        opportunities_report = {
            'date': datetime.now().isoformat(),
            'total_opportunities': len(opportunities),
            'high_conviction': [op for op in opportunities if op['score'] >= 5],
            'watch_list': [op for op in opportunities if op['score'] >= 3],
            'all_opportunities': opportunities
        }
        
        with open(f"{self.reports_dir}/opportunities.json", 'w') as f:
            json.dump(opportunities_report, f, indent=2)
        
        print(f"💰 Opportunities report saved: {len(opportunities)} opportunities")
    
    async def generate_risk_assessment(self):
        """Generate market risk assessment"""
        
        # Calculate market risk metrics
        spy_data = synthetic_ohlcv('SPY', limit=100)
        vix_data = synthetic_ohlcv('VIX', limit=100)  # Simulate VIX
        
        # Market volatility
        spy_returns = spy_data['close'].pct_change()
        market_vol = float(spy_returns.std() * np.sqrt(252) * 100)
        
        # Simulate VIX level
        current_vix = float(vix_data['close'].iloc[-1])
        
        # Risk assessment
        risk_level = "LOW"
        risk_score = 0
        alerts = []
        
        if market_vol > 20:
            risk_score += 2
            alerts.append("High market volatility detected")
        
        if current_vix > 25:
            risk_score += 3
            alerts.append("Fear index elevated")
        
        if market_vol > 25:
            risk_score += 2
            alerts.append("Extreme volatility warning")
        
        # Determine overall risk level
        if risk_score >= 5:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        risk_assessment = {
            'date': datetime.now().isoformat(),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'market_volatility': market_vol,
            'vix_level': current_vix,
            'alerts': alerts,
            'recommendations': [
                "Monitor position sizes" if risk_score >= 3 else "Normal operations",
                "Consider hedging" if risk_score >= 5 else "Standard risk management",
                "Reduce leverage" if risk_score >= 7 else "Maintain allocation"
            ]
        }
        
        with open(f"{self.reports_dir}/risk_assessment.json", 'w') as f:
            json.dump(risk_assessment, f, indent=2)
        
        print(f"⚠️ Risk assessment saved: {risk_level} risk level")


async def main():
    """Run daily reports generation"""
    generator = DailyReportsGenerator()
    await generator.generate_all_reports()


if __name__ == "__main__":
    asyncio.run(main())
