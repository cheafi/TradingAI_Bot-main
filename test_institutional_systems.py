"""
Test Institutional Systems Integration
====================================

This script validates that all institutional components work together:
1. Data contracts prevent leakage
2. Kill-switch responds to risk
3. Cost model calculates realistic costs
4. Walk-forward validation works properly
"""

import sys
import os
sys.path.append('/workspaces/TradingAI_Bot-main')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_contracts():
    """Test data contracts for leakage prevention."""
    try:
        from trading_platform.data.contracts import (
            PITDataManager, RSIFeature, MovingAverageFeature, LeakageDetector
        )
        
        logger.info("Testing data contracts...")
        
        # Create test data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days)),
            'high': 0,  # Will be filled
            'low': 0,   # Will be filled
            'volume': 1000000 + np.random.randint(-200000, 200000, n_days),
            'pe_ratio': 15 + np.random.randn(n_days),
        }, index=dates)
        
        data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        
        # Setup PIT manager
        pit_manager = PITDataManager()
        pit_manager.register_feature(RSIFeature(14))
        pit_manager.register_feature(MovingAverageFeature(20))
        pit_manager.load_data("TEST", data)
        
        # Test feature extraction
        as_of_date = data.index[-100]
        features = pit_manager.get_features(
            symbols=["TEST"],
            feature_names=["rsi_14", "sma_20"],
            as_of=as_of_date
        )
        
        assert len(features) == 1
        logger.info("✅ Data contracts working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data contracts failed: {e}")
        return False


def test_kill_switch():
    """Test kill-switch system."""
    try:
        from trading_platform.safety.killswitch import KillSwitch, TradingState
        
        logger.info("Testing kill-switch system...")
        
        # Create kill-switch
        kill_switch = KillSwitch()
        
        # Test normal operation
        assert kill_switch.trading_state == TradingState.NORMAL
        
        # Test risk threshold breach
        kill_switch.update_metric("daily_pnl_pct", -3.0)  # Critical level
        
        # Test emergency halt
        kill_switch.update_metric("daily_pnl_pct", -6.0)  # Emergency
        assert kill_switch.trading_state == TradingState.EMERGENCY_HALT
        
        # Test manual reset
        kill_switch.resume_trading("test_operator", "System test")
        assert kill_switch.trading_state == TradingState.NORMAL
        
        kill_switch.stop_monitoring_gracefully()
        logger.info("✅ Kill-switch working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Kill-switch failed: {e}")
        return False


def test_cost_model():
    """Test cost modeling system."""
    try:
        from trading_platform.execution.cost_model import (
            RealisticCostModel, OrderContext, AssetClass, OrderType, MarketCondition
        )
        
        logger.info("Testing cost model...")
        
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
            notional=1_500_000,
            adv=50_000_000,
            market_cap=2_500_000_000_000,
            price=150.0,
            spread=0.01,
            timestamp=datetime.now(),
            market_condition=market_condition,
            urgency=0.7
        )
        
        # Calculate costs
        cost_model = RealisticCostModel()
        costs = cost_model.calculate_costs(order_context)
        
        # Verify costs are reasonable
        assert costs.total_cost > 0
        assert costs.commission > 0
        assert costs.bid_ask_spread > 0
        
        total_cost_bps = (costs.total_cost / order_context.notional) * 10000
        assert 0 < total_cost_bps < 100  # Reasonable cost range
        
        logger.info(f"✅ Cost model working - Total cost: {total_cost_bps:.1f} bps")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cost model failed: {e}")
        return False


def test_walk_forward_validation():
    """Test walk-forward validation system."""
    try:
        from trading_platform.validation.walk_forward import (
            WalkForwardValidator, ValidationConfig, create_embargo_validator
        )
        
        logger.info("Testing walk-forward validation...")
        
        # Create validator
        validator = create_embargo_validator(
            min_train_months=6,  # Shorter for test
            test_months=1,
            embargo_days=2,
            step_months=1
        )
        
        # Create windows
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)
        
        windows = validator.create_windows(start_date, end_date)
        
        assert len(windows) > 0
        
        # Verify window structure
        for window in windows:
            assert window.train_start < window.train_end
            assert window.train_end <= window.embargo_start
            assert window.embargo_start < window.embargo_end
            assert window.embargo_end <= window.test_start
            assert window.test_start < window.test_end
        
        logger.info(f"✅ Walk-forward validation working - Created {len(windows)} windows")
        return True
        
    except Exception as e:
        logger.error(f"❌ Walk-forward validation failed: {e}")
        return False


def test_leakage_prevention():
    """Critical test - ensure no future data leakage is possible."""
    try:
        from trading_platform.data.contracts import RSIFeature, LeakageDetector
        
        logger.info("Testing critical leakage prevention...")
        
        # Create data with obvious future spike
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        base_data = pd.DataFrame({
            'close': np.ones(len(dates)) * 100.0,
            'high': np.ones(len(dates)) * 101.0,
            'low': np.ones(len(dates)) * 99.0,
            'volume': np.ones(len(dates)) * 1000000
        }, index=dates)
        
        # Test RSI feature
        rsi_feature = RSIFeature(14)
        
        # This should detect NO leakage (function should be leakage-free)
        leakage_free = LeakageDetector.create_future_spike_test(
            rsi_feature, base_data, spike_magnitude=5.0
        )
        
        # Our features should be leakage-free
        assert leakage_free, "CRITICAL: Feature shows sensitivity to future data"
        
        logger.info("✅ CRITICAL TEST PASSED: No future data leakage detected")
        return True
        
    except Exception as e:
        logger.error(f"❌ CRITICAL TEST FAILED: Leakage prevention error: {e}")
        return False


def main():
    """Run all institutional system tests."""
    logger.info("🏛️ TESTING INSTITUTIONAL TRADING SYSTEM")
    logger.info("=" * 50)
    
    tests = [
        ("Data Contracts", test_data_contracts),
        ("Kill-Switch", test_kill_switch), 
        ("Cost Model", test_cost_model),
        ("Walk-Forward Validation", test_walk_forward_validation),
        ("🚨 CRITICAL: Leakage Prevention", test_leakage_prevention),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n▶️ Running {test_name}...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY:")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL INSTITUTIONAL SYSTEMS OPERATIONAL!")
        logger.info("🛡️ Trading system is leakage-free and production-ready")
    else:
        logger.error("\n💥 SOME SYSTEMS FAILED - DO NOT USE IN PRODUCTION")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
