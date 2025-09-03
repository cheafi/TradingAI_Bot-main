"""
Institutional-grade configuration management using pydantic.
Type-safe, validated, environment-aware configs.
"""

from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ExchangeType(str, Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    IBKR = "ibkr"
    ALPACA = "alpaca"


class AgentType(str, Enum):
    """Available trading agents."""
    BEN_GRAHAM = "ben_graham"
    WARREN_BUFFETT = "warren_buffett"
    CHARLIE_MUNGER = "charlie_munger"
    MICHAEL_BURRY = "michael_burry"
    CATHIE_WOOD = "cathie_wood"
    PETER_LYNCH = "peter_lynch"
    STANLEY_DRUCKENMILLER = "stanley_druckenmiller"
    TECHNICALS = "technicals"
    FUNDAMENTALS = "fundamentals"
    SENTIMENT = "sentiment"
    VALUATION = "valuation"


class DataConfig(BaseModel):
    """Data source and processing configuration."""
    
    # Data vendors
    primary_vendor: str = "yfinance"
    backup_vendors: List[str] = ["alpha_vantage", "quandl"]
    
    # Point-in-time settings
    point_in_time: bool = True
    include_delisted: bool = True
    corporate_actions: bool = True
    survivorship_bias_free: bool = True
    
    # Cache and storage
    cache_dir: Path = Path("data/cache")
    database_url: Optional[str] = None
    
    # Update frequencies
    fundamental_update_hour: int = 6  # UTC
    price_update_seconds: int = 300
    
    # Quality checks
    max_missing_ratio: float = 0.1
    min_volume_threshold: float = 1000000  # USD


class BacktestConfig(BaseModel):
    """Backtesting engine configuration."""
    
    engine: str = "vectorbt"  # vectorbt, lean, nautilus, backtrader
    
    # Realism settings
    slippage_model: str = "linear"  # linear, sqrt, impact_curve
    commission_rate: float = 0.001
    borrowing_rate: float = 0.02
    
    # Implementation shortfall
    model_implementation_shortfall: bool = True
    market_impact_model: str = "almgren_chriss"
    
    # Fill simulation
    partial_fills: bool = True
    latency_ms: int = 100
    
    # Walk-forward validation
    train_period_days: int = 252  # 1 year
    test_period_days: int = 63    # 1 quarter
    min_samples: int = 1000


class RiskConfig(BaseModel):
    """Risk management configuration."""
    
    # Position limits
    max_position_size: float = 0.1  # 10% of portfolio
    max_sector_exposure: float = 0.25  # 25% in any sector
    max_gross_exposure: float = 1.0  # 100% gross
    max_net_exposure: float = 1.0   # 100% net
    
    # Volatility targeting
    target_volatility: Optional[float] = 0.15  # 15% annualized
    vol_lookback_days: int = 63
    
    # Stop losses
    max_drawdown: float = 0.05  # 5% portfolio DD
    position_stop_loss: float = 0.03  # 3% position stop
    
    # Risk metrics
    var_confidence: float = 0.95
    cvar_confidence: float = 0.99
    
    # Kill switches
    daily_loss_limit: float = 0.02  # 2% daily loss limit
    correlation_limit: float = 0.8   # Max pairwise correlation


class AgentConfig(BaseModel):
    """Individual agent configuration."""
    
    name: str
    agent_type: AgentType
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Agent-specific parameters
    parameters: Dict[str, Union[str, int, float, bool]] = {}
    
    # Risk limits
    max_positions: int = 50
    min_conviction: float = 0.6
    holding_period_days: Optional[int] = None


class PortfolioConfig(BaseModel):
    """Portfolio construction configuration."""
    
    # Allocation method
    allocation_method: str = "black_litterman"  # mv, bl, rp, hrp, equal
    
    # Rebalancing
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    rebalance_threshold: float = 0.05  # 5% drift threshold
    
    # Optimization constraints
    min_weight: float = 0.01  # 1% minimum position
    max_weight: float = 0.1   # 10% maximum position
    turnover_penalty: float = 0.001  # Transaction cost penalty
    
    # Black-Litterman specific
    tau: float = 0.025
    confidence_scaling: float = 1.0


class ExecutionConfig(BaseModel):
    """Order execution configuration."""
    
    # Default order parameters
    default_order_type: str = "market"  # market, limit, stop
    limit_price_offset: float = 0.001  # 0.1% price improvement
    
    # Execution algorithms
    use_twap: bool = True
    use_vwap: bool = False
    slice_duration_minutes: int = 15
    
    # Brokers
    primary_broker: str = "ibkr"
    backup_brokers: List[str] = ["alpaca"]
    
    # Paper trading
    paper_trading: bool = True
    paper_broker_latency: int = 1000  # ms


class TelegramConfig(BaseModel):
    """Telegram bot configuration."""
    
    token: Optional[str] = None
    chat_ids: List[int] = []
    
    # Notification settings
    send_signals: bool = True
    send_fills: bool = True
    send_risk_alerts: bool = True
    send_daily_summary: bool = True
    
    # Report timing
    daily_report_time: time = time(9, 0)  # 9 AM
    weekly_report_day: int = 5  # Friday


class SystemConfig(BaseModel):
    """Main system configuration."""
    
    environment: Environment = Environment.DEVELOPMENT
    
    # Core modules
    data: DataConfig = DataConfig()
    backtest: BacktestConfig = BacktestConfig()
    risk: RiskConfig = RiskConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    execution: ExecutionConfig = ExecutionConfig()
    telegram: TelegramConfig = TelegramConfig()
    
    # Agents
    agents: List[AgentConfig] = []
    
    # Logging
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    
    # Monitoring
    metrics_enabled: bool = True
    prometheus_port: int = 8000
    
    @validator('agents')
    def validate_agent_weights(cls, v):
        """Ensure agent weights sum to reasonable value."""
        if v:
            total_weight = sum(agent.weight for agent in v if agent.enabled)
            if total_weight > 2.0:  # Allow some flexibility
                raise ValueError("Total agent weights exceed 2.0")
        return v
    
    class Config:
        env_prefix = "TRADINGAI_"
        case_sensitive = False


def load_config(config_path: Optional[Path] = None) -> SystemConfig:
    """Load configuration from file or environment."""
    if config_path and config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return SystemConfig(**config_dict)
    
    return SystemConfig()


# Default agent configurations
DEFAULT_AGENTS = [
    AgentConfig(
        name="Ben Graham Deep Value",
        agent_type=AgentType.BEN_GRAHAM,
        weight=0.2,
        parameters={
            "min_pb_ratio": 0.7,
            "min_pe_ratio": 8.0,
            "min_current_ratio": 1.5,
            "max_debt_equity": 0.3,
            "min_fcf_yield": 0.1
        }
    ),
    AgentConfig(
        name="Warren Buffett Quality",
        agent_type=AgentType.WARREN_BUFFETT,
        weight=0.25,
        parameters={
            "min_roic": 0.15,
            "min_roe": 0.15,
            "revenue_growth_consistency": 3,  # years
            "margin_stability": 0.02,  # max std dev
            "max_debt_equity": 0.5
        }
    ),
    AgentConfig(
        name="Technical Momentum",
        agent_type=AgentType.TECHNICALS,
        weight=0.15,
        parameters={
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "ma_period_fast": 20,
            "ma_period_slow": 50,
            "volume_threshold": 1.5  # relative to average
        }
    ),
    AgentConfig(
        name="Sentiment Analysis",
        agent_type=AgentType.SENTIMENT,
        weight=0.1,
        parameters={
            "model_name": "ProsusAI/finbert",
            "sentiment_threshold": 0.7,
            "news_lookback_days": 7,
            "earning_call_weight": 2.0
        }
    )
]
