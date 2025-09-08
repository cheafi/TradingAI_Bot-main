# trading_assistant.py
from __future__ import annotations
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import List, Dict, Any, Optional

try:  # Optional import (avoid hard dependency if azure libs not installed)
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from src.config import cfg

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """Trading insight with enhanced metadata."""
    kind: str
    text: str
    meta: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    confidence: Optional[float] = None
    risk_level: Optional[str] = None


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
    """High level AI assistant facade with enhanced features.
    - If Azure OpenAI env configured and openai lib available, uses that.
    - Else falls back to lightweight heuristic summariser.
    - Includes retry logic, caching, and comprehensive error handling.
    """
    
    def __init__(self):
        self.azure_endpoint = cfg.azure_openai_endpoint
        self.deployment = cfg.azure_openai_deployment
        self.api_version = cfg.azure_openai_api_version or "2024-02-15-preview"
        self._client = None
        self._cache = {}  # Simple in-memory cache
        
        # Initialize Azure OpenAI client if available
        if self._validate_config() and OpenAI:
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
                logger.info("Azure OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Azure OpenAI client: {e}"
                )
                self._client = None
        else:
            logger.warning(
                "Azure OpenAI not available or config invalid, using fallback"
            )

    def _validate_config(self) -> bool:
        """Validate Azure OpenAI configuration."""
        required_fields = [
            self.azure_endpoint,
            self.deployment,
            self.api_version
        ]
        is_valid = all(field is not None for field in required_fields)
        if not is_valid:
            logger.warning("Azure OpenAI configuration incomplete")
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
        """Analyze trading signals with caching and fallback logic."""
        if not signals:
            return [Insight(
                kind="info",
                text="No current signals.",
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
            resp = self._client.chat.completions.create(
                model=self.deployment,
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
            "config_valid": self._validate_config(),
            "cache_stats": self.get_cache_stats(),
            "azure_endpoint": self.azure_endpoint is not None,
            "deployment": self.deployment,
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
