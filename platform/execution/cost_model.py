"""
Unified Cost Model for Institutional Trading
==========================================

This module provides comprehensive cost modeling that includes:
- Bid-ask spreads
- Market impact (temporary and permanent)
- Commission structures
- Borrowing costs
- Slippage modeling
- Implementation shortfall tracking

CRITICAL: Realistic cost modeling prevents strategy over-optimization
and ensures backtests reflect real trading conditions.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod


class OrderType(Enum):
    """Order types with different cost profiles."""
    MARKET = "market"
    LIMIT = "limit"
    MOC = "market_on_close"
    LOC = "limit_on_close"
    TWAP = "twap"
    VWAP = "vwap"
    IS = "implementation_shortfall"


class AssetClass(Enum):
    """Asset classes with different cost structures."""
    EQUITY_LARGE_CAP = "equity_large_cap"
    EQUITY_MID_CAP = "equity_mid_cap"
    EQUITY_SMALL_CAP = "equity_small_cap"
    EQUITY_INTERNATIONAL = "equity_international"
    FIXED_INCOME = "fixed_income"
    FUTURES = "futures"
    FX = "fx"
    CRYPTO = "crypto"


@dataclass
class MarketCondition:
    """Market condition parameters affecting costs."""
    volatility_regime: str  # "low", "medium", "high"
    liquidity_regime: str  # "high", "medium", "low"
    market_stress: float  # 0-1 scale
    time_of_day: str  # "open", "mid_day", "close"
    
    
@dataclass
class OrderContext:
    """Context for order execution."""
    symbol: str
    asset_class: AssetClass
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: float
    notional: float
    adv: float  # Average daily volume
    market_cap: float
    price: float
    spread: float
    timestamp: datetime
    market_condition: MarketCondition
    urgency: float = 0.5  # 0 = patient, 1 = urgent


@dataclass
class TradingCosts:
    """Comprehensive trading cost breakdown."""
    commission: float = 0.0
    bid_ask_spread: float = 0.0
    market_impact_temporary: float = 0.0
    market_impact_permanent: float = 0.0
    timing_cost: float = 0.0
    opportunity_cost: float = 0.0
    borrowing_cost: float = 0.0
    total_cost: float = 0.0
    
    def __post_init__(self):
        """Calculate total cost."""
        self.total_cost = (
            self.commission +
            self.bid_ask_spread +
            self.market_impact_temporary +
            self.market_impact_permanent +
            self.timing_cost +
            self.opportunity_cost +
            self.borrowing_cost
        )


class CostModel(ABC):
    """Abstract base class for cost models."""
    
    @abstractmethod
    def calculate_costs(self, order_context: OrderContext) -> TradingCosts:
        """Calculate trading costs for an order."""
        pass


class EquityAlmgrenChrissCostModel(CostModel):
    """
    Almgren-Chriss cost model for equity trading.
    
    This model includes:
    - Linear permanent impact
    - Square-root temporary impact
    - Risk penalty for timing
    """
    
    def __init__(self, 
                 sigma: float = 0.3,  # Annual volatility
                 gamma: float = 1e-6,  # Permanent impact coefficient
                 eta: float = 1e-5,    # Temporary impact coefficient
                 lambda_: float = 1e-3):  # Risk aversion coefficient
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        self.lambda_ = lambda_
    
    def calculate_costs(self, order_context: OrderContext) -> TradingCosts:
        """Calculate Almgren-Chriss costs."""
        # Participation rate
        participation_rate = order_context.quantity / order_context.adv
        
        # Permanent impact (linear in participation rate)
        permanent_impact = (
            self.gamma * order_context.price * 
            participation_rate * order_context.notional
        )
        
        # Temporary impact (square-root in participation rate)
        temporary_impact = (
            self.eta * order_context.price * 
            np.sqrt(participation_rate) * order_context.notional
        )
        
        # Bid-ask spread cost
        spread_cost = 0.5 * order_context.spread * order_context.notional
        
        # Risk penalty (timing cost)
        timing_cost = (
            self.lambda_ * (order_context.sigma ** 2) * 
            order_context.notional * order_context.urgency
        )
        
        return TradingCosts(
            bid_ask_spread=spread_cost,
            market_impact_temporary=temporary_impact,
            market_impact_permanent=permanent_impact,
            timing_cost=timing_cost
        )


class RealisticCostModel(CostModel):
    """
    Realistic cost model with comprehensive cost factors.
    
    This model adapts to:
    - Asset class
    - Market conditions
    - Order size and urgency
    - Time of day effects
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Commission schedules by asset class
        self.commission_rates = {
            AssetClass.EQUITY_LARGE_CAP: 0.0005,    # 5 bps
            AssetClass.EQUITY_MID_CAP: 0.0008,      # 8 bps
            AssetClass.EQUITY_SMALL_CAP: 0.0015,    # 15 bps
            AssetClass.EQUITY_INTERNATIONAL: 0.0020, # 20 bps
            AssetClass.FIXED_INCOME: 0.0010,        # 10 bps
            AssetClass.FUTURES: 0.0002,             # 2 bps
            AssetClass.FX: 0.0001,                  # 1 bp
            AssetClass.CRYPTO: 0.0025,              # 25 bps
        }
        
        # Base spreads by asset class (as % of price)
        self.base_spreads = {
            AssetClass.EQUITY_LARGE_CAP: 0.0005,
            AssetClass.EQUITY_MID_CAP: 0.0010,
            AssetClass.EQUITY_SMALL_CAP: 0.0025,
            AssetClass.EQUITY_INTERNATIONAL: 0.0015,
            AssetClass.FIXED_INCOME: 0.0020,
            AssetClass.FUTURES: 0.0002,
            AssetClass.FX: 0.0001,
            AssetClass.CRYPTO: 0.0050,
        }
        
        # Market impact coefficients
        self.impact_coefficients = {
            AssetClass.EQUITY_LARGE_CAP: {"gamma": 0.5, "eta": 0.3},
            AssetClass.EQUITY_MID_CAP: {"gamma": 0.8, "eta": 0.5},
            AssetClass.EQUITY_SMALL_CAP: {"gamma": 1.5, "eta": 1.0},
            AssetClass.EQUITY_INTERNATIONAL: {"gamma": 1.0, "eta": 0.7},
            AssetClass.FIXED_INCOME: {"gamma": 0.3, "eta": 0.2},
            AssetClass.FUTURES: {"gamma": 0.2, "eta": 0.1},
            AssetClass.FX: {"gamma": 0.1, "eta": 0.05},
            AssetClass.CRYPTO: {"gamma": 2.0, "eta": 1.5},
        }
    
    def calculate_costs(self, order_context: OrderContext) -> TradingCosts:
        """Calculate comprehensive trading costs."""
        # Base commission
        commission = self._calculate_commission(order_context)
        
        # Spread cost with market condition adjustments
        spread_cost = self._calculate_spread_cost(order_context)
        
        # Market impact
        temp_impact, perm_impact = self._calculate_market_impact(order_context)
        
        # Timing and opportunity costs
        timing_cost = self._calculate_timing_cost(order_context)
        opportunity_cost = self._calculate_opportunity_cost(order_context)
        
        # Borrowing costs (for short positions)
        borrowing_cost = self._calculate_borrowing_cost(order_context)
        
        return TradingCosts(
            commission=commission,
            bid_ask_spread=spread_cost,
            market_impact_temporary=temp_impact,
            market_impact_permanent=perm_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            borrowing_cost=borrowing_cost
        )
    
    def _calculate_commission(self, ctx: OrderContext) -> float:
        """Calculate commission costs."""
        base_rate = self.commission_rates.get(ctx.asset_class, 0.001)
        
        # Volume discounts for large orders
        volume_factor = 1.0
        if ctx.notional > 10_000_000:  # >$10M
            volume_factor = 0.7
        elif ctx.notional > 1_000_000:  # >$1M
            volume_factor = 0.85
        
        return base_rate * ctx.notional * volume_factor
    
    def _calculate_spread_cost(self, ctx: OrderContext) -> float:
        """Calculate bid-ask spread costs."""
        base_spread = self.base_spreads.get(ctx.asset_class, 0.001)
        
        # Market condition adjustments
        stress_multiplier = 1.0 + 2.0 * ctx.market_condition.market_stress
        
        # Volatility adjustment
        if ctx.market_condition.volatility_regime == "high":
            vol_multiplier = 2.0
        elif ctx.market_condition.volatility_regime == "medium":
            vol_multiplier = 1.3
        else:
            vol_multiplier = 1.0
        
        # Liquidity adjustment
        if ctx.market_condition.liquidity_regime == "low":
            liq_multiplier = 3.0
        elif ctx.market_condition.liquidity_regime == "medium":
            liq_multiplier = 1.5
        else:
            liq_multiplier = 1.0
        
        # Time of day adjustment
        if ctx.market_condition.time_of_day in ["open", "close"]:
            time_multiplier = 1.5
        else:
            time_multiplier = 1.0
        
        effective_spread = (
            base_spread * stress_multiplier * vol_multiplier * 
            liq_multiplier * time_multiplier
        )
        
        # Market orders pay full spread, limit orders pay partial
        if ctx.order_type == OrderType.MARKET:
            spread_fraction = 1.0
        elif ctx.order_type == OrderType.LIMIT:
            spread_fraction = 0.3  # Assume 30% spread capture
        else:
            spread_fraction = 0.5  # Other order types
        
        return 0.5 * effective_spread * ctx.notional * spread_fraction
    
    def _calculate_market_impact(self, ctx: OrderContext) -> Tuple[float, float]:
        """Calculate temporary and permanent market impact."""
        coeffs = self.impact_coefficients.get(
            ctx.asset_class, 
            {"gamma": 1.0, "eta": 0.5}
        )
        
        # Participation rate
        participation_rate = min(ctx.quantity / max(ctx.adv, 1), 1.0)
        
        # Size factor (square-root law)
        size_factor = np.sqrt(participation_rate)
        
        # Urgency factor
        urgency_factor = 1.0 + ctx.urgency
        
        # Market condition factor
        stress_factor = 1.0 + ctx.market_condition.market_stress
        
        # Temporary impact
        temp_impact = (
            coeffs["eta"] * size_factor * urgency_factor * 
            stress_factor * ctx.notional
        )
        
        # Permanent impact (smaller than temporary)
        perm_impact = (
            coeffs["gamma"] * participation_rate * 
            stress_factor * ctx.notional
        )
        
        return temp_impact, perm_impact
    
    def _calculate_timing_cost(self, ctx: OrderContext) -> float:
        """Calculate timing cost from delayed execution."""
        # Risk penalty for not executing immediately
        if ctx.urgency < 0.3:  # Patient orders
            base_timing_cost = 0.0001 * ctx.notional
        else:
            base_timing_cost = 0.0005 * ctx.notional * ctx.urgency
        
        # Market volatility adjustment
        if ctx.market_condition.volatility_regime == "high":
            vol_factor = 2.0
        else:
            vol_factor = 1.0
        
        return base_timing_cost * vol_factor
    
    def _calculate_opportunity_cost(self, ctx: OrderContext) -> float:
        """Calculate opportunity cost of missing execution."""
        # Cost of not trading when intended
        if ctx.order_type in [OrderType.LIMIT, OrderType.LOC]:
            # Limit orders have fill risk
            fill_probability = 0.7  # Assume 70% fill rate
            opportunity_cost = 0.0003 * ctx.notional * (1 - fill_probability)
        else:
            opportunity_cost = 0.0
        
        return opportunity_cost
    
    def _calculate_borrowing_cost(self, ctx: OrderContext) -> float:
        """Calculate borrowing costs for short positions."""
        if ctx.side == "sell" and ctx.asset_class == AssetClass.EQUITY_LARGE_CAP:
            # Assume daily borrowing cost for shorts
            daily_borrow_rate = 0.002  # 20 bps annualized
            return daily_borrow_rate * ctx.notional / 365
        
        return 0.0


class ImplementationShortfallTracker:
    """
    Track implementation shortfall for order execution analysis.
    
    Implementation shortfall measures the difference between the decision
    price and the final execution price, including all costs.
    """
    
    def __init__(self):
        self.executions: List[Dict] = []
        self.cost_model = RealisticCostModel()
    
    def record_execution(self,
                        symbol: str,
                        decision_price: float,
                        execution_price: float,
                        quantity: float,
                        side: str,
                        decision_time: datetime,
                        execution_time: datetime,
                        order_context: OrderContext):
        """Record an order execution for analysis."""
        
        # Calculate costs
        costs = self.cost_model.calculate_costs(order_context)
        
        # Calculate price impact components
        price_diff = execution_price - decision_price
        if side == "sell":
            price_diff = -price_diff  # Reverse for sells
        
        notional = quantity * decision_price
        
        execution_record = {
            "symbol": symbol,
            "decision_time": decision_time,
            "execution_time": execution_time,
            "decision_price": decision_price,
            "execution_price": execution_price,
            "quantity": quantity,
            "side": side,
            "notional": notional,
            "price_diff": price_diff,
            "delay_minutes": (execution_time - decision_time).total_seconds() / 60,
            "costs": costs,
            "implementation_shortfall_bps": (price_diff / decision_price) * 10000,
            "total_cost_bps": (costs.total_cost / notional) * 10000
        }
        
        self.executions.append(execution_record)
    
    def get_analysis(self, 
                    symbol: Optional[str] = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict:
        """Get implementation shortfall analysis."""
        
        # Filter executions
        filtered_executions = self.executions
        
        if symbol:
            filtered_executions = [e for e in filtered_executions if e["symbol"] == symbol]
        
        if start_date:
            filtered_executions = [e for e in filtered_executions 
                                 if e["execution_time"] >= start_date]
        
        if end_date:
            filtered_executions = [e for e in filtered_executions 
                                 if e["execution_time"] <= end_date]
        
        if not filtered_executions:
            return {"error": "No executions found for criteria"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(filtered_executions)
        
        # Calculate statistics
        stats = {
            "total_executions": len(df),
            "total_notional": df["notional"].sum(),
            "avg_implementation_shortfall_bps": df["implementation_shortfall_bps"].mean(),
            "std_implementation_shortfall_bps": df["implementation_shortfall_bps"].std(),
            "avg_total_cost_bps": df["total_cost_bps"].mean(),
            "avg_delay_minutes": df["delay_minutes"].mean(),
            "cost_breakdown": {
                "commission_bps": (df.apply(lambda x: x["costs"].commission, axis=1).sum() / 
                                 df["notional"].sum()) * 10000,
                "spread_bps": (df.apply(lambda x: x["costs"].bid_ask_spread, axis=1).sum() / 
                             df["notional"].sum()) * 10000,
                "market_impact_bps": (df.apply(lambda x: x["costs"].market_impact_temporary + 
                                              x["costs"].market_impact_permanent, axis=1).sum() / 
                                     df["notional"].sum()) * 10000,
            }
        }
        
        return stats


class CostModelFactory:
    """Factory for creating appropriate cost models."""
    
    @staticmethod
    def create_cost_model(model_type: str = "realistic") -> CostModel:
        """Create a cost model of the specified type."""
        if model_type == "realistic":
            return RealisticCostModel()
        elif model_type == "almgren_chriss":
            return EquityAlmgrenChrissCostModel()
        else:
            raise ValueError(f"Unknown cost model type: {model_type}")


# Global implementation shortfall tracker
_global_is_tracker: Optional[ImplementationShortfallTracker] = None


def get_is_tracker() -> ImplementationShortfallTracker:
    """Get the global implementation shortfall tracker."""
    global _global_is_tracker
    if _global_is_tracker is None:
        _global_is_tracker = ImplementationShortfallTracker()
    return _global_is_tracker


if __name__ == "__main__":
    # Example usage
    
    # Create order context
    market_condition = MarketCondition(
        volatility_regime="medium",
        liquidity_regime="high",
        market_stress=0.2,
        time_of_day="mid_day"
    )
    
    order_context = OrderContext(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY_LARGE_CAP,
        order_type=OrderType.MARKET,
        side="buy",
        quantity=10000,
        notional=1_500_000,  # $1.5M
        adv=50_000_000,      # $50M ADV
        market_cap=2_500_000_000_000,  # $2.5T
        price=150.0,
        spread=0.01,
        timestamp=datetime.now(),
        market_condition=market_condition,
        urgency=0.7
    )
    
    # Calculate costs
    cost_model = RealisticCostModel()
    costs = cost_model.calculate_costs(order_context)
    
    print("Trading Cost Analysis:")
    print(f"Commission: ${costs.commission:,.2f}")
    print(f"Bid-Ask Spread: ${costs.bid_ask_spread:,.2f}")
    print(f"Temporary Impact: ${costs.market_impact_temporary:,.2f}")
    print(f"Permanent Impact: ${costs.market_impact_permanent:,.2f}")
    print(f"Timing Cost: ${costs.timing_cost:,.2f}")
    print(f"Total Cost: ${costs.total_cost:,.2f}")
    print(f"Total Cost (bps): {(costs.total_cost/order_context.notional)*10000:.1f}")
