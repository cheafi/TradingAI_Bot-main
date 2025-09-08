# trading_assistant.py - Multi-Asset AI Trading Assistant
from __future__ import annotations
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import List, Dict, Any, Optional

try:  # Optional import (avoid hard dependency if azure libs not installed)
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from src.config import cfg

# Configure logging
logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Supported asset classes for multi-asset trading."""
    CRYPTO = "crypto"
    FX = "fx"
    GOLD = "gold"
    US_EQUITY = "us_equity"
    HK_EQUITY = "hk_equity"


@dataclass
class AssetInfo:
    """Asset-specific information for trading."""
    symbol: str
    asset_class: AssetClass
    exchange: str
    timezone: str
    trading_hours: str
    cost_model: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """Standardized trading signal across all asset classes."""
    # Common fields (通用欄位)
    symbol: str
    asset_class: AssetClass
    direction: str  # "long" or "short"
    entry_range: Dict[str, float]  # {"low": X, "high": Y}
    targets: Dict[str, float]  # {"t1": X, "t2": Y}
    stop_loss: float
    horizon: str  # "intraday", "swing", "position"
    conviction: int  # 1-5 scale
    expected_alpha_bps: float  # Net of all costs
    reasoning: List[str]  # 3 key reasons
    suggested_size_pct: float  # % of portfolio
    
    # Asset-specific fields (資產特有欄位)
    asset_specific: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    
    def get_hkt_time(self) -> str:
        """Get signal time in HKT format."""
        import datetime
        dt = datetime.datetime.fromtimestamp(self.timestamp)
        # Convert to HKT (UTC+8)
        hkt_dt = dt + datetime.timedelta(hours=8)
        return hkt_dt.strftime("%Y-%m-%d %H:%M:%S HKT")


@dataclass
class Insight:
    """Trading insight with enhanced metadata for multi-asset support."""
    kind: str
    text: str
    meta: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    asset_class: Optional[AssetClass] = None
    
    def get_bilingual_text(self) -> Dict[str, str]:
        """Get bilingual text for the insight."""
        # This would be enhanced with proper translation
        return {
            "en": self.text,
            "zh": self.text  # TODO: Add Chinese translations
        }


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.warning(
                            f"Final retry failed for {func.__name__}: {e}"
                        )
                        raise
                    wait_time = delay * (2 ** attempt)
                    func_name = func.__name__
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func_name}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


class TradingAssistant:
    """Multi-Asset AI Trading Assistant with enhanced features.
    
    Supports: Crypto • FX • Gold • US & HK Equities
    Features:
    - Asset-aware signal generation and analysis
    - Bilingual support (English + 繁中)
    - Cost model integration per asset class
    - Risk management across asset classes
    - Azure OpenAI integration with fallback
    """
    
    # Asset class configurations
    ASSET_CONFIGS = {
        AssetClass.CRYPTO: {
            "trading_hours": "24/7",
            "cost_fields": ["maker_fee", "taker_fee", "funding_rate"],
            "timezone": "UTC",
            "min_edge_multiplier": 2.0
        },
        AssetClass.FX: {
            "trading_hours": "24/5",
            "cost_fields": ["spread_pips", "swap_rate"],
            "timezone": "UTC",
            "min_edge_multiplier": 2.0
        },
        AssetClass.GOLD: {
            "trading_hours": "23/5",
            "cost_fields": ["exchange_fee", "spread", "roll_cost"],
            "timezone": "UTC",
            "min_edge_multiplier": 2.0
        },
        AssetClass.US_EQUITY: {
            "trading_hours": "09:30-16:00 EST",
            "cost_fields": ["commission", "spread", "borrow_fee"],
            "timezone": "EST",
            "min_edge_multiplier": 2.0
        },
        AssetClass.HK_EQUITY: {
            "trading_hours": "09:30-12:00,13:00-16:00 HKT",
            "cost_fields": ["commission", "stamp_duty", "levy", "spread"],
            "timezone": "HKT",
            "min_edge_multiplier": 2.0
        }
    }
    
    def __init__(self):
        # OpenAI Configuration (supports both Standard and Azure)
        self.openai_api_key = cfg.openai_api_key
        self.openai_model = cfg.openai_model
        self.openai_base_url = cfg.openai_base_url
        
        # Azure OpenAI Configuration (legacy)
        self.azure_endpoint = cfg.azure_openai_endpoint
        self.deployment = cfg.azure_openai_deployment
        self.api_version = cfg.azure_openai_api_version or "2024-02-15-preview"
        
        self._client = None
        self._cache = {}  # Simple in-memory cache
        
        # Portfolio limits
        self.max_single_position = 0.06  # 6%
        self.max_daily_turnover = 0.15   # 15%
        self.target_portfolio_vol = 0.12  # 12%
        
        # Initialize OpenAI client (try standard OpenAI first, then Azure)
        if self._init_standard_openai():
            logger.info("Standard OpenAI client initialized successfully")
        elif self._init_azure_openai():
            logger.info("Azure OpenAI client initialized successfully")
        else:
            logger.warning(
                "No OpenAI API available, using fallback mode"
            )

    def _init_standard_openai(self) -> bool:
        """Initialize standard OpenAI client."""
        if not OpenAI or not self.openai_api_key:
            return False
            
        try:
            self._client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            # Test the connection with a simple call
            self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize standard OpenAI: {e}")
            self._client = None
            return False

    def _init_azure_openai(self) -> bool:
        """Initialize Azure OpenAI client (legacy support)."""
        if not OpenAI or not self._validate_azure_config():
            return False
            
        try:
            base_url = (
                f"{self.azure_endpoint}/openai/deployments/"
                f"{self.deployment}"
            )
            api_key = os.getenv(
                "AZURE_OPENAI_KEY",
                os.getenv("AZURE_OPENAI_API_KEY")
            )
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI: {e}")
            self._client = None
            return False

    def _validate_azure_config(self) -> bool:
        """Validate Azure OpenAI configuration."""
        required_fields = [
            self.azure_endpoint,
            self.deployment,
            self.api_version
        ]
        is_valid = all(field is not None for field in required_fields)
        if not is_valid:
            logger.debug("Azure OpenAI configuration incomplete")
        return is_valid

    def _get_cache_key(self, signals: List[Dict[str, Any]]) -> str:
        """Generate cache key for signals."""
        # Sort signals by symbol for consistent caching
        sorted_signals = sorted(signals, key=lambda x: x.get('symbol', ''))
        content = str(sorted_signals)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[List[Insight]]:
        """Get cached result if available and not expired."""
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            # Cache expires after 5 minutes
            if time.time() - timestamp < 300:
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return cached_data
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        return None

    def _set_cached_result(
        self, cache_key: str, result: List[Insight]
    ) -> None:
        """Cache the result with timestamp."""
        self._cache[cache_key] = (result, time.time())
        logger.debug(f"Cached result for key: {cache_key[:8]}...")

    # ---------------------- internal helpers ----------------------
    def _llm_available(self) -> bool:
        return self._client is not None

    # ---------------------- public API ----------------------------
    def analyse_signals(self, signals: List[Dict[str, Any]]) -> List[Insight]:
        """Analyze trading signals with multi-asset awareness and caching."""
        if not signals:
            return [Insight(
                kind="info",
                text="No current signals. | 當前無交易訊號。",
                meta={},
                confidence=1.0,
                risk_level="low"
            )]
        
        # Check cache first
        cache_key = self._get_cache_key(signals)
        if cached_result := self._get_cached_result(cache_key):
            return cached_result
        
        # Generate new analysis
        if self._llm_available():
            result = self._analyse_with_llm(signals)
        else:
            result = self._heuristic(signals)
        
        # Cache the result
        self._set_cached_result(cache_key, result)
        return result

    def analyse_multi_asset_signals(
        self, signals: List[TradingSignal]
    ) -> List[Insight]:
        """Analyze trading signals across multiple asset classes."""
        if not signals:
            return [Insight(
                kind="info",
                text="No multi-asset signals. | 無跨資產訊號。",
                meta={},
                confidence=1.0,
                risk_level="low"
            )]
        
        insights = []
        
        # Group signals by asset class
        signals_by_asset = {}
        for signal in signals:
            asset_class = signal.asset_class
            if asset_class not in signals_by_asset:
                signals_by_asset[asset_class] = []
            signals_by_asset[asset_class].append(signal)
        
        # Analyze each asset class
        for asset_class, asset_signals in signals_by_asset.items():
            asset_insights = self._analyse_asset_class_signals(
                asset_class, asset_signals
            )
            insights.extend(asset_insights)
        
        # Add cross-asset analysis
        cross_asset_insights = self._analyse_cross_asset_correlation(signals)
        insights.extend(cross_asset_insights)
        
        # Add portfolio-level analysis
        portfolio_insights = self._analyse_portfolio_level(signals)
        insights.extend(portfolio_insights)
        
        return insights

    def _analyse_asset_class_signals(
        self, asset_class: AssetClass, signals: List[TradingSignal]
    ) -> List[Insight]:
        """Analyze signals within a specific asset class."""
        if not signals:
            return []
            
        # Filter signals by edge >= 2x cost rule
        valid_signals = [
            s for s in signals
            if s.expected_alpha_bps >= (
                self.ASSET_CONFIGS[asset_class]["min_edge_multiplier"] *
                self._estimate_total_cost(s)
            )
        ]
        
        if not valid_signals:
            return [Insight(
                kind="warning",
                text=f"No {asset_class.value} signals meet edge≥2×cost rule",
                meta={
                    "asset_class": asset_class.value,
                    "filtered_count": len(signals)
                },
                confidence=0.8,
                risk_level="medium",
                asset_class=asset_class
            )]
        
        # Generate asset-specific insights
        insights = []
        
        # Signal count and quality
        valid_count = len(valid_signals)
        avg_conviction = sum(s.conviction for s in valid_signals) / valid_count
        avg_alpha = sum(
            s.expected_alpha_bps for s in valid_signals
        ) / valid_count
        
        insights.append(Insight(
            kind="asset_summary",
            text=f"{asset_class.value}: {len(valid_signals)} signals, "
                 f"avg conviction {avg_conviction:.1f}, "
                 f"avg alpha {avg_alpha:.0f}bps",
            meta={
                "asset_class": asset_class.value,
                "signal_count": len(valid_signals),
                "avg_conviction": avg_conviction,
                "avg_alpha_bps": avg_alpha
            },
            confidence=0.9,
            risk_level="medium",
            asset_class=asset_class
        ))
        
        # Best signal in class
        best_signal = max(
            valid_signals,
            key=lambda s: s.expected_alpha_bps * s.conviction
        )
        insights.append(Insight(
            kind="best_signal",
            text=(
                f"Top {asset_class.value}: {best_signal.symbol} "
                f"{best_signal.direction} "
                f"(α:{best_signal.expected_alpha_bps:.0f}bps, "
                f"conv:{best_signal.conviction})"
            ),
            meta={
                "asset_class": asset_class.value,
                "symbol": best_signal.symbol,
                "direction": best_signal.direction,
                "alpha_bps": best_signal.expected_alpha_bps,
                "conviction": best_signal.conviction
            },
            confidence=best_signal.confidence or 0.7,
            risk_level=best_signal.risk_level or "medium",
            asset_class=asset_class
        ))
        
        return insights

    def _analyse_cross_asset_correlation(
        self, signals: List[TradingSignal]
    ) -> List[Insight]:
        """Analyze correlations and interactions across asset classes."""
        if len(signals) < 2:
            return []
            
        # Simple correlation analysis
        asset_classes = set(s.asset_class for s in signals)
        
        if len(asset_classes) > 1:
            classes_text = ', '.join(ac.value for ac in asset_classes)
            return [Insight(
                kind="diversification",
                text=(
                    f"Multi-asset exposure across {len(asset_classes)} "
                    f"classes: {classes_text}"
                ),
                meta={
                    "asset_classes": [ac.value for ac in asset_classes],
                    "diversification_score": len(asset_classes) / 5.0
                },
                confidence=0.8,
                risk_level="low"
            )]
        
        return []

    def _analyse_portfolio_level(
        self, signals: List[TradingSignal]
    ) -> List[Insight]:
        """Analyze portfolio-level metrics and constraints."""
        if not signals:
            return []
            
        insights = []
        
        # Total suggested exposure
        total_exposure = sum(s.suggested_size_pct for s in signals)
        
        if total_exposure > 1.0:  # 100%
            insights.append(Insight(
                kind="risk_warning",
                text=f"⚠️ Total exposure {total_exposure:.1%} exceeds 100%",
                meta={"total_exposure": total_exposure},
                confidence=0.9,
                risk_level="high"
            ))
        
        # Check single position limits
        position_warnings = [
            Insight(
                kind="position_limit",
                text=(
                    f"⚠️ {signal.symbol} position "
                    f"{signal.suggested_size_pct:.1%} "
                    f"exceeds {self.max_single_position:.1%} limit"
                ),
                meta={
                    "symbol": signal.symbol,
                    "suggested_size": signal.suggested_size_pct,
                    "limit": self.max_single_position
                },
                confidence=0.9,
                risk_level="high"
            )
            for signal in signals
            if signal.suggested_size_pct > self.max_single_position
        ]
        insights.extend(position_warnings)
        
        # Portfolio alpha estimate
        total_alpha = sum(
            s.expected_alpha_bps * s.suggested_size_pct for s in signals
        )
        
        insights.append(Insight(
            kind="portfolio_alpha",
            text=f"Portfolio expected alpha: {total_alpha:.0f}bps | "
                 f"組合預期超額：{total_alpha:.0f}基點",
            meta={
                "total_alpha_bps": total_alpha,
                "signal_count": len(signals)
            },
            confidence=0.7,
            risk_level="medium"
        ))
        
        return insights

    def _estimate_total_cost(self, signal: TradingSignal) -> float:
        """Estimate total trading cost in basis points for a signal."""
        # This would be enhanced with actual cost models per asset class
        base_cost = {
            AssetClass.CRYPTO: 10,      # 10 bps base
            AssetClass.FX: 5,           # 5 bps base
            AssetClass.GOLD: 8,         # 8 bps base
            AssetClass.US_EQUITY: 12,   # 12 bps base
            AssetClass.HK_EQUITY: 25,   # 25 bps base (stamp duty)
        }
        
        return base_cost.get(signal.asset_class, 10)

    @retry_on_failure(max_retries=2, delay=0.5)
    def _analyse_with_llm(
        self, signals: List[Dict[str, Any]]
    ) -> List[Insight]:  # pragma: no cover - network
        """Analyze signals using Azure OpenAI with enhanced prompting."""
        if self._client is None:
            # Fallback to heuristic if client not available
            return self._heuristic(signals)
            
        # Enhanced prompt with better structure
        signal_summary = self._format_signals_for_llm(signals)
        prompt = (
            "You are an expert trading assistant. Analyze these signals and "
            "provide a concise summary (max 80 words).\n\n"
            f"Signals:\n{signal_summary}\n\n"
            "Include:\n"
            "- Overall market sentiment\n"
            "- Key opportunities and risks\n"
            "- Risk warning if high volatility (>3% ATR)\n"
            "- Confidence level (high/medium/low)"
        )
        
        try:
            # Use appropriate model based on client type
            model = (
                self.deployment if self.azure_endpoint
                else self.openai_model
            )
            
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional trading assistant."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
            )
            
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
                
            txt = content.strip()
            # Extract confidence and risk level from response
            confidence, risk_level = self._parse_llm_response(txt)
            
            return [
                Insight(
                    kind="summary",
                    text=txt,
                    meta={
                        "source": "azure_openai",
                        "signals_count": len(signals)
                    },
                    confidence=confidence,
                    risk_level=risk_level
                )
            ]
                
        except Exception as e:  # Fail soft
            logger.error(f"LLM analysis failed: {e}")
            error_insight = Insight(
                kind="error",
                text=f"LLM error: {e}",
                meta={"error_type": type(e).__name__},
                confidence=0.0,
                risk_level="unknown"
            )
            return [error_insight] + self._heuristic(signals)

    def _format_signals_for_llm(self, signals: List[Dict[str, Any]]) -> str:
        """Format signals in a structured way for LLM analysis."""
        formatted = []
        for i, signal in enumerate(signals[:10], 1):  # Limit to top 10
            symbol = signal.get('symbol', 'Unknown')
            direction = signal.get('direction', 'unknown')
            confidence = signal.get('confidence', 0.0)
            reason = signal.get('reason', 'No reason provided')
            
            formatted.append(
                f"{i}. {symbol} - {direction.upper()} "
                f"(confidence: {confidence:.2f}) - {reason}"
            )
        
        if len(signals) > 10:
            formatted.append(f"... and {len(signals) - 10} more signals")
            
        return "\n".join(formatted)

    def _parse_llm_response(
        self, response: str
    ) -> tuple[Optional[float], Optional[str]]:
        """Extract confidence and risk level from LLM response."""
        response_lower = response.lower()
        
        # Parse confidence
        confidence = None
        if "high confidence" in response_lower:
            confidence = 0.8
        elif "medium confidence" in response_lower:
            confidence = 0.6
        elif "low confidence" in response_lower:
            confidence = 0.4
            
        # Parse risk level
        risk_level = None
        high_risk_words = ["high risk", "very risky", "dangerous"]
        if any(word in response_lower for word in high_risk_words):
            risk_level = "high"
        elif any(
            word in response_lower for word in ["medium risk", "moderate"]
        ):
            risk_level = "medium"
        elif any(
            word in response_lower
            for word in ["low risk", "safe", "stable"]
        ):
            risk_level = "low"
            
        return confidence, risk_level

    def _get_best_signal(
        self, signals: List[Dict[str, Any]], signal_type: str
    ) -> Optional[Insight]:
        """Get the best signal of a given type (long/short)."""
        if not signals:
            return None
            
        best_signal = max(signals, key=lambda x: x.get('confidence', 0))
        symbol = best_signal.get('symbol', '?')
        confidence = best_signal.get('confidence', 0.0)
        
        return Insight(
            kind=f"focus_{signal_type}",
            text=f"Best {signal_type}: {symbol} (conf: {confidence:.2f})",
            meta=best_signal,
            confidence=confidence,
            risk_level="medium"
        )

    def _heuristic(self, signals: List[Dict[str, Any]]) -> List[Insight]:
        """Enhanced heuristic analysis with better insights."""
        longs = [s for s in signals if s.get("direction") == "long"]
        shorts = [s for s in signals if s.get("direction") == "short"]
        
        # Calculate average confidence
        avg_confidence = sum(
            s.get('confidence', 0.5) for s in signals
        ) / len(signals) if signals else 0.0
        
        # Determine risk level based on signal distribution
        risk_level = "medium"
        if len(signals) > 20:
            risk_level = "high"  # Too many signals might indicate noise
        elif len(signals) < 3:
            risk_level = "low"   # Few signals, lower risk
            
        count_text = (
            f"Longs: {len(longs)} Shorts: {len(shorts)} "
            f"Total: {len(signals)} (avg confidence: {avg_confidence:.2f})"
        )
        out = [Insight(
            kind="count",
            text=count_text,
            meta={
                "signal_distribution": {
                    "longs": len(longs),
                    "shorts": len(shorts)
                }
            },
            confidence=avg_confidence,
            risk_level=risk_level
        )]
        
        # Add best signals if available
        if best_long := self._get_best_signal(longs, "long"):
            out.append(best_long)
            
        if best_short := self._get_best_signal(shorts, "short"):
            out.append(best_short)
            
        # Add market sentiment insight
        if len(longs) > len(shorts) * 2:
            sentiment = "bullish"
        elif len(shorts) > len(longs) * 2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
            
        ratio = len(longs) / max(len(shorts), 1)
        out.append(Insight(
            kind="sentiment",
            text=f"Market sentiment: {sentiment}",
            meta={"sentiment": sentiment, "ratio": ratio},
            confidence=avg_confidence,
            risk_level=risk_level
        ))
        
        return out

    def clear_cache(self) -> None:
        """Clear the insights cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        active_entries = 0
        expired_entries = 0
        
        for _, (_, timestamp) in self._cache.items():
            if current_time - timestamp < 300:  # 5 minutes
                active_entries += 1
            else:
                expired_entries += 1
                
        return {
            "total_entries": len(self._cache),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "cache_hit_potential": active_entries / max(len(self._cache), 1)
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the assistant."""
        return {
            "llm_available": self._llm_available(),
            "openai_configured": bool(self.openai_api_key),
            "azure_configured": self._validate_azure_config(),
            "cache_stats": self.get_cache_stats(),
            "openai_model": self.openai_model,
            "azure_deployment": self.deployment,
            "api_version": self.api_version
        }


# Global singleton instance
_assistant_singleton: Optional[TradingAssistant] = None


def get_trading_assistant() -> TradingAssistant:
    """Get the singleton trading assistant instance."""
    global _assistant_singleton
    if _assistant_singleton is None:
        _assistant_singleton = TradingAssistant()
        logger.info("TradingAssistant singleton initialized")
    return _assistant_singleton


def reset_trading_assistant() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _assistant_singleton
    if _assistant_singleton:
        _assistant_singleton.clear_cache()
    _assistant_singleton = None
    logger.info("TradingAssistant singleton reset")


# For backward compatibility
assistant_singleton = get_trading_assistant()
