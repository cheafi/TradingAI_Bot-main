import logging
from typing import List, Dict
from src.agents.damodaran_agent import DamodaranAgent
from src.agents.graham_agent import GrahamAgent
from src.agents.trend_following_agent import TrendFollowingAgent
from src.agents.mean_reversion_agent import MeanReversionAgent
import pandas as pd

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Manages multiple trading agents and consolidates their signals.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.agents: Dict[str, List[object]] = {}
        for symbol in symbols:
            self.agents[symbol] = [
                DamodaranAgent(symbol),
                GrahamAgent(symbol),
                TrendFollowingAgent(symbol),
                MeanReversionAgent(symbol)
            ]

    def generate_signals(self) -> Dict[str, Dict[str, str]]:
        """Generate trading signals from all agents for all symbols."""
        signals = {}
        for symbol, agent_list in self.agents.items():
            signals[symbol] = {}
            for agent in agent_list:
                try:
                    signal = agent.generate_signal()
                    signals[symbol][agent.__class__.__name__] = signal
                except Exception as e:
                    logger.error(f"Error generating signal for {agent.__class__.__name__} on {symbol}: {e}")
                    signals[symbol][agent.__class__.__name__] = "HOLD"  # Default to HOLD on error
        return signals

    def consolidate_signals(self, signals: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """Consolidate signals from multiple agents into a single signal per symbol."""
        consolidated_signals = {}
        for symbol, agent_signals in signals.items():
            # Implement consolidation logic here (e.g., majority voting, weighted averaging)
            # This is a placeholder and needs to be implemented with actual consolidation logic
            consolidated_signals[symbol] = "HOLD"  # Default to HOLD for now
        return consolidated_signals

    def detect_bias(self, signals: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """Detect bias in agent signals."""
        bias_metrics = {}
        for symbol, agent_signals in signals.items():
            bias_metrics[symbol] = {}
            for agent_name, signal in agent_signals.items():
                # Implement bias detection logic here
                # Check if the agent consistently favors a particular signal
                # This is a simplified example and needs to be refined
                if signal == "BUY" or signal == "STRONG BUY":
                    bias_metrics[symbol][agent_name] = 0.1  # Assign a small bias score
                else:
                    bias_metrics[symbol][agent_name] = 0.0
        return bias_metrics

    def mitigate_bias(self, signals: Dict[str, Dict[str, str]], bias_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, str]]:
        """Mitigate bias in agent signals."""
        mitigated_signals = {}
        for symbol, agent_signals in signals.items():
            mitigated_signals[symbol] = {}
            for agent_name, signal in agent_signals.items():
                # Implement bias mitigation logic here
                # This is a placeholder and needs to be implemented with actual bias mitigation logic
                mitigated_signals[symbol][agent_name] = signal  # Return the original signal for now
        return mitigated_signals
