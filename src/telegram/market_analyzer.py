"""Real-time Market Analysis and Opportunity Detection for Telegram Bot"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
import talib
import requests

logger = logging.getLogger(__name__)

@dataclass
class MarketOpportunity:
    """Data class for market opportunities"""
    symbol: str
    action: str  # "BUY" or "SHORT"
    current_price: float
    target_price: float
    stop_loss: float
    max_drop: float
    confidence_level: float  # 0-100
    reason: str
    event: str
    setup_type: str
    curve_analysis: str
    timestamp: datetime
    market_cap: Optional[float] = None
    volume: Optional[float] = None
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None

@dataclass
class PortfolioPosition:
    """Data class for portfolio positions"""
    id: int
    symbol: str
    action: str
    entry_price: float
    current_price: float
    quantity: float
    target_price: float
    stop_loss: float
    confidence_level: float
    entry_date: datetime
    status: str  # "ACTIVE", "CLOSED", "STOPPED"
    pnl: float = 0.0
    pnl_percent: float = 0.0

class MarketAnalyzer:
    """Advanced market analysis and opportunity detection"""
    
    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = db_path
        self.init_database()
        self.watchlist = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
            "SPY", "QQQ", "IWM", "DIA", "VTI", "ARKK", "SOXL", "TQQQ",
            "BTCUSD=X", "ETHUSD=X", "EURUSD=X", "GBPUSD=X"
        ]
        
    def init_database(self):
        """Initialize SQLite database for portfolio tracking"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    quantity REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    entry_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'ACTIVE',
                    pnl REAL DEFAULT 0.0,
                    pnl_percent REAL DEFAULT 0.0,
                    reason TEXT,
                    event TEXT,
                    setup_type TEXT
                )
            """)
            
            # Create opportunities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    max_drop REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    reason TEXT,
                    event TEXT,
                    setup_type TEXT,
                    curve_analysis TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'PENDING'
                )
            """)
            
            conn.commit()

    async def analyze_market_opportunities(self) -> List[MarketOpportunity]:
        """Analyze market for buy/sell opportunities"""
        opportunities = []
        
        for symbol in self.watchlist:
            try:
                opportunity = await self._analyze_symbol(symbol)
                if opportunity and opportunity.confidence_level > 60:
                    opportunities.append(opportunity)
                    await self._save_opportunity(opportunity)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                
        return sorted(opportunities, key=lambda x: x.confidence_level, reverse=True)

    async def _analyze_symbol(self, symbol: str) -> Optional[MarketOpportunity]:
        """Comprehensive analysis of a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period="60d", interval="1d")
            if len(hist) < 20:
                return None
                
            current_price = hist['Close'].iloc[-1]
            
            # Technical analysis
            technical_signals = self._technical_analysis(hist)
            
            # Fundamental analysis
            fundamental_signals = await self._fundamental_analysis(ticker, symbol)
            
            # Market sentiment
            sentiment_signals = await self._sentiment_analysis(symbol)
            
            # Combine all signals
            opportunity = self._combine_signals(
                symbol, current_price, technical_signals, 
                fundamental_signals, sentiment_signals
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def _technical_analysis(self, hist: pd.DataFrame) -> Dict:
        """Perform technical analysis on price data"""
        close = hist['Close'].values
        high = hist['High'].values
        low = hist['Low'].values
        volume = hist['Volume'].values
        
        signals = {}
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)
        signals['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        signals['macd_bullish'] = macd[-1] > macd_signal[-1] if not np.isnan(macd[-1]) else False
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
        signals['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
        
        # Moving averages
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        signals['ma_trend'] = "BULLISH" if sma_20[-1] > sma_50[-1] else "BEARISH"
        
        # Support/Resistance
        signals['support'] = np.min(low[-20:])
        signals['resistance'] = np.max(high[-20:])
        
        # Volume analysis
        avg_volume = np.mean(volume[-20:])
        signals['volume_spike'] = volume[-1] > avg_volume * 1.5
        
        return signals

    async def _fundamental_analysis(self, ticker, symbol: str) -> Dict:
        """Analyze fundamental metrics"""
        signals = {}
        
        try:
            info = ticker.info
            
            # Market cap
            signals['market_cap'] = info.get('marketCap', 0)
            
            # P/E ratio
            pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
            signals['pe_ratio'] = pe_ratio
            signals['pe_attractive'] = 5 < pe_ratio < 25 if pe_ratio else False
            
            # Revenue growth
            signals['revenue_growth'] = info.get('revenueGrowth', 0)
            
            # Debt to equity
            signals['debt_to_equity'] = info.get('debtToEquity', 0)
            
            # Analyst recommendations
            signals['recommendation'] = info.get('recommendationKey', 'hold')
            
        except Exception as e:
            logger.error(f"Fundamental analysis error for {symbol}: {e}")
            
        return signals

    async def _sentiment_analysis(self, symbol: str) -> Dict:
        """Analyze market sentiment and news"""
        signals = {}
        
        try:
            # News sentiment (simplified)
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if news:
                # Simple sentiment scoring based on news titles
                positive_words = ['up', 'gain', 'bullish', 'strong', 'beat', 'exceed', 'growth']
                negative_words = ['down', 'loss', 'bearish', 'weak', 'miss', 'decline', 'fall']
                
                sentiment_score = 0
                for article in news[:5]:  # Check recent 5 articles
                    title = article.get('title', '').lower()
                    sentiment_score += sum(1 for word in positive_words if word in title)
                    sentiment_score -= sum(1 for word in negative_words if word in title)
                
                signals['news_sentiment'] = sentiment_score
            else:
                signals['news_sentiment'] = 0
                
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            signals['news_sentiment'] = 0
            
        return signals

    def _combine_signals(self, symbol: str, current_price: float, 
                        technical: Dict, fundamental: Dict, sentiment: Dict) -> Optional[MarketOpportunity]:
        """Combine all signals to generate trading opportunity"""
        
        confidence = 0
        action = None
        reason_parts = []
        
        # Technical analysis scoring
        if technical.get('rsi', 50) < 30:  # Oversold
            confidence += 20
            reason_parts.append("RSI oversold")
            action = "BUY"
        elif technical.get('rsi', 50) > 70:  # Overbought
            confidence += 15
            reason_parts.append("RSI overbought")
            action = "SHORT"
            
        if technical.get('macd_bullish', False):
            confidence += 15
            reason_parts.append("MACD bullish crossover")
            if not action:
                action = "BUY"
        
        if technical.get('ma_trend') == "BULLISH":
            confidence += 10
            reason_parts.append("Moving average uptrend")
            if not action:
                action = "BUY"
        elif technical.get('ma_trend') == "BEARISH":
            confidence += 10
            reason_parts.append("Moving average downtrend")
            if not action:
                action = "SHORT"
                
        if technical.get('volume_spike', False):
            confidence += 10
            reason_parts.append("High volume spike")
            
        # Fundamental scoring
        if fundamental.get('pe_attractive', False):
            confidence += 10
            reason_parts.append("Attractive P/E ratio")
            
        if fundamental.get('revenue_growth', 0) > 0.1:
            confidence += 10
            reason_parts.append("Strong revenue growth")
            
        # Sentiment scoring
        news_sentiment = sentiment.get('news_sentiment', 0)
        if news_sentiment > 0:
            confidence += min(news_sentiment * 5, 15)
            reason_parts.append("Positive news sentiment")
        elif news_sentiment < 0:
            confidence += min(abs(news_sentiment) * 5, 15)
            reason_parts.append("Negative news sentiment")
            if not action:
                action = "SHORT"
        
        # Set default action if none determined
        if not action:
            action = "BUY" if confidence > 50 else "SHORT"
            
        # Calculate target and stop loss
        if action == "BUY":
            target_price = current_price * 1.08  # 8% target
            stop_loss = current_price * 0.95     # 5% stop loss
            max_drop = 5.0
        else:  # SHORT
            target_price = current_price * 0.92  # 8% target down
            stop_loss = current_price * 1.05     # 5% stop loss
            max_drop = 5.0
            
        # Generate event and setup description
        event = self._get_market_event(symbol, technical, fundamental)
        setup_type = self._get_setup_type(technical)
        curve_analysis = self._get_curve_analysis(technical)
        
        if confidence < 60:  # Minimum confidence threshold
            return None
            
        return MarketOpportunity(
            symbol=symbol,
            action=action,
            current_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            max_drop=max_drop,
            confidence_level=min(confidence, 95),  # Cap at 95%
            reason=" | ".join(reason_parts),
            event=event,
            setup_type=setup_type,
            curve_analysis=curve_analysis,
            timestamp=datetime.now(),
            rsi=technical.get('rsi'),
            macd_signal="BULLISH" if technical.get('macd_bullish') else "BEARISH"
        )

    def _get_market_event(self, symbol: str, technical: Dict, fundamental: Dict) -> str:
        """Determine the market event driving the opportunity"""
        events = []
        
        if technical.get('volume_spike'):
            events.append("High volume breakout")
        if technical.get('rsi', 50) < 30:
            events.append("Oversold bounce setup")
        if technical.get('rsi', 50) > 70:
            events.append("Overbought correction setup")
        if fundamental.get('revenue_growth', 0) > 0.2:
            events.append("Strong earnings growth")
            
        return events[0] if events else "Technical pattern recognition"

    def _get_setup_type(self, technical: Dict) -> str:
        """Identify the technical setup type"""
        if technical.get('bb_position', 0.5) < 0.1:
            return "Bollinger Band squeeze"
        elif technical.get('bb_position', 0.5) > 0.9:
            return "Bollinger Band breakout"
        elif technical.get('rsi', 50) < 30:
            return "RSI oversold reversal"
        elif technical.get('rsi', 50) > 70:
            return "RSI overbought reversal"
        else:
            return "Moving average trend following"

    def _get_curve_analysis(self, technical: Dict) -> str:
        """Analyze price curve patterns"""
        rsi = technical.get('rsi', 50)
        bb_pos = technical.get('bb_position', 0.5)
        
        if rsi < 30 and bb_pos < 0.2:
            return "Double bottom formation with oversold conditions"
        elif rsi > 70 and bb_pos > 0.8:
            return "Double top formation with overbought conditions"
        elif technical.get('ma_trend') == "BULLISH":
            return "Ascending triangle with bullish momentum"
        else:
            return "Consolidation pattern with pending breakout"

    async def _save_opportunity(self, opportunity: MarketOpportunity):
        """Save opportunity to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO opportunities (
                    symbol, action, current_price, target_price, stop_loss,
                    max_drop, confidence_level, reason, event, setup_type,
                    curve_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                opportunity.symbol, opportunity.action, opportunity.current_price,
                opportunity.target_price, opportunity.stop_loss, opportunity.max_drop,
                opportunity.confidence_level, opportunity.reason, opportunity.event,
                opportunity.setup_type, opportunity.curve_analysis
            ))
            conn.commit()

    async def add_position(self, symbol: str, action: str, quantity: float, 
                          opportunity: MarketOpportunity) -> int:
        """Add new position to portfolio"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO positions (
                    symbol, action, entry_price, current_price, quantity,
                    target_price, stop_loss, confidence_level, reason,
                    event, setup_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, action, opportunity.current_price, opportunity.current_price,
                quantity, opportunity.target_price, opportunity.stop_loss,
                opportunity.confidence_level, opportunity.reason,
                opportunity.event, opportunity.setup_type
            ))
            conn.commit()
            return cursor.lastrowid

    async def update_positions(self) -> List[PortfolioPosition]:
        """Update all active positions with current prices"""
        positions = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE status = 'ACTIVE'")
            
            for row in cursor.fetchall():
                position = PortfolioPosition(*row)
                
                # Update current price
                try:
                    ticker = yf.Ticker(position.symbol)
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    
                    # Calculate P&L
                    if position.action == "BUY":
                        pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                    else:  # SHORT
                        pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
                    
                    pnl = pnl_percent * position.quantity * position.entry_price / 100
                    
                    # Check stop loss or target
                    status = position.status
                    if position.action == "BUY":
                        if current_price <= position.stop_loss:
                            status = "STOPPED"
                        elif current_price >= position.target_price:
                            status = "TARGET_REACHED"
                    else:  # SHORT
                        if current_price >= position.stop_loss:
                            status = "STOPPED"
                        elif current_price <= position.target_price:
                            status = "TARGET_REACHED"
                    
                    # Update database
                    cursor.execute("""
                        UPDATE positions 
                        SET current_price = ?, pnl = ?, pnl_percent = ?, status = ?
                        WHERE id = ?
                    """, (current_price, pnl, pnl_percent, status, position.id))
                    
                    # Update position object
                    position.current_price = current_price
                    position.pnl = pnl
                    position.pnl_percent = pnl_percent
                    position.status = status
                    
                    positions.append(position)
                    
                except Exception as e:
                    logger.error(f"Error updating position {position.symbol}: {e}")
            
            conn.commit()
            
        return positions

    async def get_portfolio_summary(self) -> Dict:
        """Get overall portfolio summary"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total P&L
            cursor.execute("SELECT SUM(pnl), AVG(pnl_percent) FROM positions WHERE status != 'PENDING'")
            total_pnl, avg_return = cursor.fetchone()
            
            # Active positions count
            cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'ACTIVE'")
            active_positions = cursor.fetchone()[0]
            
            # Win rate
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
                    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losses,
                    COUNT(*) as total
                FROM positions WHERE status IN ('STOPPED', 'TARGET_REACHED', 'CLOSED')
            """)
            wins, losses, total_trades = cursor.fetchone()
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_pnl': total_pnl or 0,
                'avg_return': avg_return or 0,
                'active_positions': active_positions,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'wins': wins,
                'losses': losses
            }

    async def get_us_market_sentiment(self) -> Dict:
        """Get overall US market sentiment"""
        try:
            # Get major indices
            indices = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'DIA': 'Dow Jones',
                'IWM': 'Russell 2000'
            }
            
            market_data = {}
            overall_sentiment = 0
            
            for symbol, name in indices.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if len(hist) >= 2:
                    change_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                    market_data[name] = {
                        'change_pct': change_pct,
                        'price': hist['Close'].iloc[-1]
                    }
                    overall_sentiment += change_pct
            
            # VIX (Fear index)
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="2d")
            vix_level = vix_data['Close'].iloc[-1] if len(vix_data) > 0 else 20
            
            # Sentiment interpretation
            if overall_sentiment > 1:
                sentiment = "BULLISH"
            elif overall_sentiment < -1:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
                
            return {
                'sentiment': sentiment,
                'overall_change': overall_sentiment / len(indices),
                'vix_level': vix_level,
                'indices': market_data,
                'fear_greed': "FEAR" if vix_level > 25 else "GREED" if vix_level < 15 else "NEUTRAL"
            }
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {'sentiment': 'UNKNOWN', 'error': str(e)}
