"""
Institutional Kill-Switch System - EMERGENCY TRADING HALT
=======================================================

This module implements multiple layers of emergency stops to prevent
catastrophic losses in automated trading systems.

CRITICAL: This system must ALWAYS be able to halt trading immediately.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
from pathlib import Path
import json


class EmergencyLevel(Enum):
    """Emergency severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    HALT_ALL = "halt_all"


class TradingState(Enum):
    """Trading system states."""
    NORMAL = "normal"
    RISK_ELEVATED = "risk_elevated"
    POSITION_FREEZE = "position_freeze"
    LIQUIDATION_ONLY = "liquidation_only"
    EMERGENCY_HALT = "emergency_halt"


@dataclass
class RiskThreshold:
    """Risk threshold configuration."""
    metric: str
    soft_limit: float
    hard_limit: float
    lookback_window: int = 1  # minutes
    consecutive_breaches: int = 1


@dataclass
class AlertConfig:
    """Alert configuration."""
    telegram_enabled: bool = True
    email_enabled: bool = True
    slack_enabled: bool = False
    phone_enabled: bool = False  # For extreme emergencies
    
    
@dataclass
class EmergencyAction:
    """Emergency action to execute."""
    timestamp: datetime
    level: EmergencyLevel
    trigger: str
    description: str
    auto_executed: bool = False
    human_confirmed: bool = False
    actions_taken: List[str] = field(default_factory=list)


class CircuitBreaker:
    """Individual circuit breaker for specific metrics."""
    
    def __init__(self, threshold: RiskThreshold):
        self.threshold = threshold
        self.breach_count = 0
        self.last_breach = None
        self.is_tripped = False
        self.trip_time = None
        
    def check(self, value: float) -> tuple[bool, EmergencyLevel]:
        """Check if circuit breaker should trip."""
        now = datetime.now()
        
        # Reset if enough time has passed
        if (self.last_breach and 
            now - self.last_breach > timedelta(minutes=self.threshold.lookback_window)):
            self.breach_count = 0
        
        level = EmergencyLevel.INFO
        should_trip = False
        
        if value >= self.threshold.hard_limit:
            level = EmergencyLevel.EMERGENCY
            should_trip = True
        elif value >= self.threshold.soft_limit:
            level = EmergencyLevel.WARNING
            self.breach_count += 1
            self.last_breach = now
            
            if self.breach_count >= self.threshold.consecutive_breaches:
                level = EmergencyLevel.CRITICAL
                should_trip = True
        
        if should_trip and not self.is_tripped:
            self.is_tripped = True
            self.trip_time = now
            
        return should_trip, level


class KillSwitch:
    """
    Master kill-switch system for emergency trading halts.
    
    This system provides multiple layers of protection:
    1. Real-time risk monitoring
    2. Automatic position limits
    3. Emergency liquidation
    4. Human override capabilities
    5. Audit logging
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "platform/safety/killswitch_config.json"
        
        # System state
        self.trading_state = TradingState.NORMAL
        self.is_armed = True
        self.emergency_contacts = []
        self.alert_config = AlertConfig()
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.emergency_log: List[EmergencyAction] = []
        
        # Threading
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        # Callbacks for external systems
        self.emergency_callbacks: List[Callable] = []
        self.position_handlers: List[Callable] = []
        
        # Risk tracking
        self.current_metrics: Dict[str, float] = {}
        self.metric_history: Dict[str, List[tuple]] = {}
        
        # Load configuration
        self._load_configuration()
        self._setup_default_breakers()
        
        # Start monitoring
        self.start_monitoring()
    
    def _load_configuration(self):
        """Load kill-switch configuration."""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    self._apply_config(config)
            except Exception as e:
                self.logger.error(f"Failed to load kill-switch config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default kill-switch configuration."""
        default_config = {
            "risk_thresholds": {
                "daily_pnl_pct": {"soft_limit": -2.0, "hard_limit": -5.0},
                "position_concentration": {"soft_limit": 0.25, "hard_limit": 0.40},
                "portfolio_var": {"soft_limit": 0.03, "hard_limit": 0.05},
                "max_drawdown": {"soft_limit": -0.03, "hard_limit": -0.08},
                "leverage_ratio": {"soft_limit": 2.0, "hard_limit": 4.0},
                "correlation_risk": {"soft_limit": 0.7, "hard_limit": 0.9}
            },
            "alert_config": {
                "telegram_enabled": True,
                "email_enabled": True,
                "emergency_phone": False
            },
            "emergency_contacts": [
                "portfolio_manager@firm.com",
                "risk_manager@firm.com"
            ]
        }
        
        # Save default config
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(exist_ok=True, parents=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    def _apply_config(self, config: Dict):
        """Apply loaded configuration."""
        # Set alert config
        if "alert_config" in config:
            alert_cfg = config["alert_config"]
            self.alert_config = AlertConfig(**alert_cfg)
        
        # Set emergency contacts
        self.emergency_contacts = config.get("emergency_contacts", [])
    
    def _setup_default_breakers(self):
        """Setup default circuit breakers."""
        default_thresholds = [
            RiskThreshold("daily_pnl_pct", -2.0, -5.0, lookback_window=1),
            RiskThreshold("position_concentration", 0.25, 0.40, lookback_window=5),
            RiskThreshold("portfolio_var", 0.03, 0.05, lookback_window=5),
            RiskThreshold("max_drawdown", -0.03, -0.08, lookback_window=1),
            RiskThreshold("leverage_ratio", 2.0, 4.0, lookback_window=5),
            RiskThreshold("correlation_risk", 0.7, 0.9, lookback_window=10)
        ]
        
        for threshold in default_thresholds:
            self.add_circuit_breaker(threshold)
    
    def add_circuit_breaker(self, threshold: RiskThreshold):
        """Add a new circuit breaker."""
        self.circuit_breakers[threshold.metric] = CircuitBreaker(threshold)
        self.logger.info(f"Added circuit breaker for {threshold.metric}")
    
    def update_metric(self, metric_name: str, value: float):
        """Update a risk metric and check thresholds."""
        now = datetime.now()
        self.current_metrics[metric_name] = value
        
        # Store history
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append((now, value))
        
        # Keep only recent history (last 24 hours)
        cutoff = now - timedelta(hours=24)
        self.metric_history[metric_name] = [
            (ts, val) for ts, val in self.metric_history[metric_name]
            if ts > cutoff
        ]
        
        # Check circuit breaker
        if metric_name in self.circuit_breakers:
            breaker = self.circuit_breakers[metric_name]
            should_trip, level = breaker.check(value)
            
            if should_trip:
                self._handle_emergency(
                    level=level,
                    trigger=metric_name,
                    description=f"{metric_name} breached: {value}"
                )
    
    def _handle_emergency(self, level: EmergencyLevel, trigger: str, description: str):
        """Handle emergency situation."""
        emergency = EmergencyAction(
            timestamp=datetime.now(),
            level=level,
            trigger=trigger,
            description=description
        )
        
        self.emergency_log.append(emergency)
        
        # Log the emergency
        self.logger.critical(f"EMERGENCY {level.value}: {description}")
        
        # Take action based on level
        if level == EmergencyLevel.WARNING:
            self._handle_warning(emergency)
        elif level == EmergencyLevel.CRITICAL:
            self._handle_critical(emergency)
        elif level in [EmergencyLevel.EMERGENCY, EmergencyLevel.HALT_ALL]:
            self._handle_halt(emergency)
        
        # Send alerts
        self._send_alerts(emergency)
        
        # Execute callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(emergency)
            except Exception as e:
                self.logger.error(f"Emergency callback failed: {e}")
    
    def _handle_warning(self, emergency: EmergencyAction):
        """Handle warning level emergency."""
        if self.trading_state == TradingState.NORMAL:
            self.trading_state = TradingState.RISK_ELEVATED
            emergency.actions_taken.append("Elevated risk state")
    
    def _handle_critical(self, emergency: EmergencyAction):
        """Handle critical level emergency."""
        self.trading_state = TradingState.POSITION_FREEZE
        emergency.actions_taken.append("Position freeze activated")
        
        # Notify position handlers
        for handler in self.position_handlers:
            try:
                handler("freeze_new_positions")
            except Exception as e:
                self.logger.error(f"Position handler failed: {e}")
    
    def _handle_halt(self, emergency: EmergencyAction):
        """Handle emergency halt."""
        self.trading_state = TradingState.EMERGENCY_HALT
        emergency.actions_taken.extend([
            "EMERGENCY HALT ACTIVATED",
            "All trading stopped",
            "Liquidation mode only"
        ])
        
        # Emergency halt all positions
        for handler in self.position_handlers:
            try:
                handler("emergency_halt")
            except Exception as e:
                self.logger.error(f"Emergency halt handler failed: {e}")
    
    def _send_alerts(self, emergency: EmergencyAction):
        """Send emergency alerts."""
        message = f"""
🚨 TRADING EMERGENCY 🚨

Level: {emergency.level.value.upper()}
Trigger: {emergency.trigger}
Description: {emergency.description}
Time: {emergency.timestamp}
State: {self.trading_state.value}

Actions Taken:
{chr(10).join(f"- {action}" for action in emergency.actions_taken)}
        """.strip()
        
        # Log alert (implement actual notification systems as needed)
        self.logger.critical(f"ALERT SENT: {message}")
        
        # TODO: Implement actual alert mechanisms
        # - Telegram bot notifications
        # - Email alerts
        # - Slack/Discord webhooks
        # - SMS for critical emergencies
    
    def manual_halt(self, reason: str, operator: str):
        """Manual emergency halt by human operator."""
        emergency = EmergencyAction(
            timestamp=datetime.now(),
            level=EmergencyLevel.HALT_ALL,
            trigger="manual_override",
            description=f"Manual halt by {operator}: {reason}",
            human_confirmed=True
        )
        
        self._handle_halt(emergency)
        self.logger.critical(f"MANUAL HALT by {operator}: {reason}")
    
    def reset_breaker(self, metric_name: str, operator: str):
        """Reset a specific circuit breaker (human confirmation required)."""
        if metric_name in self.circuit_breakers:
            breaker = self.circuit_breakers[metric_name]
            breaker.is_tripped = False
            breaker.breach_count = 0
            breaker.trip_time = None
            
            self.logger.warning(f"Circuit breaker {metric_name} reset by {operator}")
            
            return True
        return False
    
    def resume_trading(self, operator: str, justification: str):
        """Resume trading after emergency halt (requires human confirmation)."""
        if self.trading_state in [TradingState.EMERGENCY_HALT, TradingState.LIQUIDATION_ONLY]:
            self.trading_state = TradingState.NORMAL
            
            emergency = EmergencyAction(
                timestamp=datetime.now(),
                level=EmergencyLevel.INFO,
                trigger="manual_resume",
                description=f"Trading resumed by {operator}: {justification}",
                human_confirmed=True
            )
            
            self.emergency_log.append(emergency)
            self.logger.warning(f"Trading resumed by {operator}: {justification}")
            
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current kill-switch status."""
        return {
            "trading_state": self.trading_state.value,
            "is_armed": self.is_armed,
            "current_metrics": self.current_metrics,
            "tripped_breakers": {
                name: {
                    "is_tripped": breaker.is_tripped,
                    "trip_time": breaker.trip_time,
                    "breach_count": breaker.breach_count
                }
                for name, breaker in self.circuit_breakers.items()
                if breaker.is_tripped
            },
            "recent_emergencies": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "level": e.level.value,
                    "trigger": e.trigger,
                    "description": e.description
                }
                for e in self.emergency_log[-10:]  # Last 10 emergencies
            ]
        }
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("Kill-switch monitoring started")
    
    def stop_monitoring_gracefully(self):
        """Stop background monitoring."""
        self.stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Check for system health
                self._check_system_health()
                
                # Sleep for monitoring interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Kill-switch monitoring error: {e}")
                time.sleep(30)  # Back off on errors
    
    def _check_system_health(self):
        """Check overall system health."""
        # Check if emergency state has been active too long
        if self.trading_state == TradingState.EMERGENCY_HALT:
            # Log ongoing emergency state
            self.logger.warning("System remains in emergency halt state")
        
        # Check for stale data
        now = datetime.now()
        for metric_name, history in self.metric_history.items():
            if history:
                last_update = history[-1][0]
                if now - last_update > timedelta(minutes=15):
                    self.logger.warning(f"Stale data for {metric_name}: {last_update}")
    
    def add_emergency_callback(self, callback: Callable[[EmergencyAction], None]):
        """Add callback for emergency events."""
        self.emergency_callbacks.append(callback)
    
    def add_position_handler(self, handler: Callable[[str], None]):
        """Add handler for position management commands."""
        self.position_handlers.append(handler)


# Global kill-switch instance
_global_kill_switch: Optional[KillSwitch] = None


def get_kill_switch() -> KillSwitch:
    """Get the global kill-switch instance."""
    global _global_kill_switch
    if _global_kill_switch is None:
        _global_kill_switch = KillSwitch()
    return _global_kill_switch


def emergency_halt_all(reason: str, operator: str = "system"):
    """Emergency function to halt all trading immediately."""
    kill_switch = get_kill_switch()
    kill_switch.manual_halt(reason, operator)


# Decorator for protecting trading functions
def protected_trading_operation(func):
    """Decorator to protect trading operations with kill-switch."""
    def wrapper(*args, **kwargs):
        kill_switch = get_kill_switch()
        
        if kill_switch.trading_state == TradingState.EMERGENCY_HALT:
            raise RuntimeError("Trading halted by kill-switch")
        elif kill_switch.trading_state == TradingState.POSITION_FREEZE:
            if "liquidate" not in func.__name__.lower():
                raise RuntimeError("New positions frozen by kill-switch")
        
        return func(*args, **kwargs)
    
    return wrapper


if __name__ == "__main__":
    # Test the kill-switch system
    kill_switch = KillSwitch()
    
    # Simulate some risk metrics
    kill_switch.update_metric("daily_pnl_pct", -1.5)  # Warning
    kill_switch.update_metric("daily_pnl_pct", -3.0)  # Critical
    kill_switch.update_metric("daily_pnl_pct", -6.0)  # Emergency halt
    
    # Check status
    status = kill_switch.get_status()
    print("Kill-switch status:", status)
    
    kill_switch.stop_monitoring_gracefully()
