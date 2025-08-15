import numpy as np
import pandas as pd
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def optimize(self, returns: pd.DataFrame, target_return: float = 0.2) -> Dict[str, float]:
        """Optimize portfolio weights using cvxpy."""
        try:
            # Calculate mean returns and covariance matrix
            mu = np.array(returns.mean())
            sigma = returns.cov().to_numpy()

            # Define variables
            weights = cp.Variable(returns.shape[1])

            # Define constraints
            ret = mu @ weights
            risk = cp.quad_form(weights, sigma)
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,  # No short selling
                ret >= target_return  # Target return constraint
            ]

            # Define objective function (maximize Sharpe ratio)
            objective = cp.Minimize(risk)

            # Solve the problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status == cp.OPTIMAL:
                return dict(zip(returns.columns, weights.value))
            else:
                logger.warning(f"Optimization failed: {problem.status}")
                return {col: 1.0 / returns.shape[1] for col in returns.columns}  # Equal weights if optimization fails
        except Exception as e:
            logger.error(f"Error during portfolio optimization: {e}")
            return {col: 1.0 / returns.shape[1] for col in returns.columns}  # Equal weights on error
