"""
Multi-Agent Trading System - Main Integration
============================================

This script ties together all components of the multi-agent trading system:
- Legendary investor agents
- Decision gateway for signal ensemble
- Agent onboarding and evaluation
- Real-time dashboard

Launch this to run the complete multi-agent trading shop.
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append('/workspaces/TradingAI_Bot-main')

# Import all our components
try:
    from apps.agents.legendary_investors import (
        LEGENDARY_AGENTS, create_agent, MarketRegime
    )
    from apps.agents.decision_gateway import DecisionGateway
    from apps.agents.agent_onboarding import AgentOnboardingSystem
    from apps.agents.multi_agent_dashboard import MultiAgentDashboard
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all agent modules are available.")
    sys.exit(1)


class MultiAgentTradingSystem:
    """
    Complete multi-agent trading system orchestrator.
    
    Manages the full lifecycle:
    1. Agent initialization and onboarding
    2. Signal generation and ensemble
    3. Portfolio optimization and execution
    4. Performance monitoring and agent promotion/demotion
    5. Real-time dashboard and reporting
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Core components
        self.decision_gateway = DecisionGateway()
        self.onboarding_system = AgentOnboardingSystem()
        self.dashboard = MultiAgentDashboard()
        
        # System state
        self.is_running = False
        self.current_positions = {}
        self.performance_history = []
        
        self.logger.info("🚀 Multi-Agent Trading System initialized")
    
    def _setup_logging(self):
        """Setup system logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/multi_agent_system.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("multi_agent_system")
    
    def initialize_system(self):
        """Initialize the complete trading system."""
        
        print("🎯 Initializing Multi-Agent Trading System")
        print("=" * 60)
        
        # 1. Initialize core agents
        print("\n1️⃣ Initializing Core Agents...")
        core_agents = [
            "simons_stat_arb",
            "dalio_macro", 
            "ptj_trend",
            "microstructure",
            "drawdown_guardian"
        ]
        
        for agent_type in core_agents:
            try:
                agent = create_agent(agent_type)
                if agent:
                    # Start onboarding process
                    success = self.onboarding_system.onboard_new_agent(agent_type)
                    if success:
                        print(f"   ✅ {agent_type}: Onboarded to sandbox")
                    else:
                        print(f"   ⚠️ {agent_type}: Already exists or failed")
                else:
                    print(f"   ❌ {agent_type}: Failed to create")
            except Exception as e:
                print(f"   ❌ {agent_type}: Error - {e}")
        
        # 2. Setup market data
        print("\n2️⃣ Setting up Market Data...")
        self.market_data = self._generate_sample_market_data()
        print(f"   ✅ Generated data for {len(self.market_data.columns)} symbols")
        print(f"   📅 Date range: {self.market_data.index[0]} to {self.market_data.index[-1]}")
        
        # 3. Initialize portfolio
        print("\n3️⃣ Initializing Portfolio...")
        self.universe = list(self.market_data.columns)
        print(f"   ✅ Trading universe: {len(self.universe)} symbols")
        
        # 4. System health check
        print("\n4️⃣ System Health Check...")
        agent_status = self.decision_gateway.get_agent_status()
        active_agents = len([a for a in agent_status.values() if a['is_active']])
        print(f"   ✅ Active agents: {active_agents}")
        print(f"   ✅ Decision gateway: Ready")
        print(f"   ✅ Onboarding system: Ready")
        
        self.is_running = True
        print("\n🎉 System initialization complete!")
        
        return True
    
    def _generate_sample_market_data(self):
        """Generate sample market data for demonstration."""
        # Create realistic market data
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=365)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Mix of large cap stocks
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA'
        ]
        
        # Generate correlated returns with different volatilities
        np.random.seed(42)
        
        # Base market factor
        market_returns = np.random.normal(0.0008, 0.012, len(dates))
        
        # Individual stock data
        data = {}
        for i, symbol in enumerate(symbols):
            # Stock-specific parameters
            beta = np.random.uniform(0.7, 1.5)
            alpha = np.random.normal(0.0002, 0.0001)
            idiosync_vol = np.random.uniform(0.015, 0.025)
            
            # Generate returns
            stock_returns = (
                alpha +
                beta * market_returns +
                np.random.normal(0, idiosync_vol, len(dates))
            )
            
            # Convert to price series
            initial_price = np.random.uniform(50, 500)
            prices = initial_price * np.cumprod(1 + stock_returns)
            data[symbol] = prices
        
        return pd.DataFrame(data, index=dates)
    
    def run_trading_session(self, session_date: datetime = None):
        """Run a single trading session."""
        
        if not self.is_running:
            self.logger.error("System not initialized. Call initialize_system() first.")
            return False
        
        if session_date is None:
            session_date = datetime.now()
        
        self.logger.info(f"🔄 Running trading session for {session_date.date()}")
        
        try:
            # 1. Generate portfolio signals
            portfolio_signals = self.decision_gateway.generate_portfolio_signals(
                self.market_data, self.universe, session_date
            )
            
            if portfolio_signals:
                self.logger.info(f"📊 Generated {len(portfolio_signals)} portfolio signals")
                
                # Log signals
                for signal in portfolio_signals[:5]:  # Show first 5
                    self.logger.info(
                        f"   🎯 {signal.ticker}: {signal.direction} "
                        f"{signal.expected_alpha_bps:.1f}bps "
                        f"(conf: {signal.confidence:.2f})"
                    )
            else:
                self.logger.warning("⚠️ No signals generated")
            
            # 2. Portfolio construction (simplified)
            new_positions = {}
            for signal in portfolio_signals:
                new_positions[signal.ticker] = {
                    'weight': signal.target_weight,
                    'direction': signal.direction,
                    'expected_alpha': signal.expected_alpha_bps,
                    'agents': signal.contributing_agents
                }
            
            # 3. Risk check (simplified)
            total_weight = sum([pos['weight'] for pos in new_positions.values()])
            if total_weight > 1.0:
                self.logger.warning(f"⚠️ Portfolio overweight: {total_weight:.1%}")
            
            # 4. Update positions
            self.current_positions = new_positions
            
            # 5. Performance evaluation (if we have previous positions)
            if hasattr(self, 'previous_positions') and self.previous_positions:
                self._evaluate_performance(session_date)
            
            self.previous_positions = new_positions.copy()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading session: {e}")
            return False
    
    def _evaluate_performance(self, session_date: datetime):
        """Evaluate performance and update agent KPIs."""
        
        # Calculate returns for each position (simplified)
        actual_returns = {}
        costs = {}
        
        for ticker in self.previous_positions.keys():
            if ticker in self.market_data.columns:
                # Simplified return calculation
                actual_returns[ticker] = np.random.normal(0.001, 0.02)  # Random return
                costs[ticker] = 0.0008  # 8 bps cost
        
        # Update agent performance tracking
        self.decision_gateway.update_agent_performance(
            actual_returns, costs, session_date
        )
        
        # Store performance history
        self.performance_history.append({
            'date': session_date,
            'positions': len(self.current_positions),
            'total_alpha': sum([pos['expected_alpha'] for pos in self.current_positions.values()]),
            'total_weight': sum([pos['weight'] for pos in self.current_positions.values()])
        })
    
    def run_agent_evaluations(self):
        """Run periodic agent evaluations."""
        
        self.logger.info("📊 Running agent evaluations...")
        
        evaluations = self.onboarding_system.run_periodic_evaluations(
            self.market_data, datetime.now()
        )
        
        if evaluations:
            self.logger.info(f"📈 Completed {len(evaluations)} agent evaluations")
            
            for evaluation in evaluations:
                self.logger.info(
                    f"   {evaluation.agent_name}: {evaluation.recommendation} "
                    f"(Score: {evaluation.overall_score:.1f})"
                )
        else:
            self.logger.info("   No evaluations due at this time")
        
        return evaluations
    
    def get_system_status(self):
        """Get comprehensive system status."""
        
        agent_roster = self.onboarding_system.get_agent_roster()
        agent_status = self.decision_gateway.get_agent_status()
        
        status = {
            'is_running': self.is_running,
            'current_regime': self.decision_gateway.current_regime.value,
            'total_agents': len(agent_roster),
            'active_agents': len([a for a in agent_roster.values() if a['is_active']]),
            'current_positions': len(self.current_positions),
            'total_weight': sum([pos['weight'] for pos in self.current_positions.values()]),
            'agent_details': agent_roster,
            'recent_performance': self.performance_history[-10:] if self.performance_history else []
        }
        
        return status
    
    def launch_dashboard(self):
        """Launch the interactive dashboard."""
        
        self.logger.info("🚀 Launching Multi-Agent Dashboard...")
        
        try:
            # This would typically launch Streamlit
            print("\n" + "="*60)
            print("🎛️ MULTI-AGENT TRADING DASHBOARD")
            print("="*60)
            print("\n📊 To launch the interactive dashboard, run:")
            print("   streamlit run apps/agents/multi_agent_dashboard.py")
            print("\n🌐 Dashboard will be available at: http://localhost:8501")
            print("\n" + "="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error launching dashboard: {e}")
            return False
    
    def demo_system(self):
        """Run a complete demonstration of the system."""
        
        print("\n🎭 MULTI-AGENT TRADING SYSTEM DEMONSTRATION")
        print("=" * 60)
        
        # 1. Initialize
        if not self.initialize_system():
            print("❌ Failed to initialize system")
            return
        
        # 2. Run a few trading sessions
        print("\n📅 Running Sample Trading Sessions...")
        for i in range(5):
            session_date = datetime.now() - pd.Timedelta(days=4-i)
            success = self.run_trading_session(session_date)
            if success:
                print(f"   ✅ Day {i+1}: Session completed")
            else:
                print(f"   ❌ Day {i+1}: Session failed")
        
        # 3. Run evaluations
        print("\n🎯 Running Agent Evaluations...")
        evaluations = self.run_agent_evaluations()
        
        # 4. Show system status
        print("\n📊 Current System Status:")
        status = self.get_system_status()
        
        print(f"   🎛️ System Running: {status['is_running']}")
        print(f"   🌍 Market Regime: {status['current_regime']}")
        print(f"   🤖 Total Agents: {status['total_agents']}")
        print(f"   ✅ Active Agents: {status['active_agents']}")
        print(f"   💼 Current Positions: {status['current_positions']}")
        print(f"   📈 Portfolio Weight: {status['total_weight']:.1%}")
        
        # 5. Show agent roster
        print(f"\n👥 Agent Roster:")
        for agent_name, info in status['agent_details'].items():
            status_emoji = {"active": "🟢", "sandbox": "🟡", "veteran": "⭐", "retired": "🔴"}.get(info['status'], "⚪")
            print(f"   {status_emoji} {agent_name}: {info['status']} | {info['allocation']:.1%}")
        
        # 6. Performance summary
        if status['recent_performance']:
            print(f"\n📈 Recent Performance Summary:")
            recent = status['recent_performance'][-1]
            print(f"   🎯 Latest Alpha: {recent['total_alpha']:.1f} bps")
            print(f"   💼 Portfolio Utilization: {recent['total_weight']:.1%}")
            print(f"   📊 Number of Positions: {recent['positions']}")
        
        # 7. Dashboard launch instructions
        self.launch_dashboard()
        
        print("\n🎉 Demonstration completed successfully!")
        print("\n💡 Next Steps:")
        print("   1. Launch the dashboard for real-time monitoring")
        print("   2. Add more legendary investor agents")
        print("   3. Connect to live market data")
        print("   4. Integrate with execution systems")
        print("   5. Deploy to production environment")


def main():
    """Main entry point."""
    
    # ASCII Art Header
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║    🤖 MULTI-AGENT TRADING SYSTEM                         ║
    ║                                                           ║
    ║    Legendary Investors AI Trading Shop                   ║
    ║    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                     ║
    ║                                                           ║
    ║    🎯 Decision Gateway + Agent Onboarding               ║
    ║    📊 ROI-Focused Measurement                           ║
    ║    🚀 Plug-and-Play Agent Architecture                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize and run system
    system = MultiAgentTradingSystem()
    
    # Run full demonstration
    try:
        system.demo_system()
    except KeyboardInterrupt:
        print("\n\n🛑 System stopped by user")
    except Exception as e:
        print(f"\n\n❌ System error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Multi-Agent Trading System shutdown complete")


if __name__ == "__main__":
    main()
