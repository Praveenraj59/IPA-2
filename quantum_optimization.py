import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(assets, predicted_returns, risk_tolerance):
    """
    Optimize portfolio using classical Markowitz Mean-Variance Optimization.
    """
    # Define the objective function (negative Sharpe ratio)
    def objective(weights):
        portfolio_return = np.dot(weights, predicted_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(predicted_returns), weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio  # Minimize negative Sharpe ratio

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(len(assets)))

    # Initial guess: equal weights
    initial_guess = np.ones(len(assets)) / len(assets)

    # Optimize
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x  # Return optimized weights
    else:
        raise ValueError("Optimization failed")