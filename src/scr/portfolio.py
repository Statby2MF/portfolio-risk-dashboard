"""
Gestion de portefeuille : allocation optimale, frontière efficiente
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_portfolio_returns(returns_df, weights):
    """
    Calcule les rendements du portefeuille
    """
    return returns_df.dot(weights)


def calculate_portfolio_volatility(returns_df, weights):
    """
    Calcule la volatilité du portefeuille
    """
    cov_matrix = returns_df.cov() * 252  # Annualisé
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return vol


def calculate_portfolio_return(returns_df, weights):
    """
    Calcule le rendement annualisé du portefeuille
    """
    return np.sum(returns_df.mean() * weights) * 252


def calculate_sharpe(returns_df, weights, risk_free_rate=0.02):
    """
    Calcule le ratio de Sharpe du portefeuille
    """
    ret = calculate_portfolio_return(returns_df, weights)
    vol = calculate_portfolio_volatility(returns_df, weights)
    sharpe = (ret - risk_free_rate) / vol
    return sharpe


def optimize_portfolio(returns_df, target_return=None):
    """
    Optimise le portefeuille (maximise le Sharpe)
    """
    n_assets = len(returns_df.columns)
    
    # Contraintes
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Somme des poids = 1
    ]
    
    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: calculate_portfolio_return(returns_df, w) - target_return
        })
    
    # Bornes (pas de short)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Point de départ : équipondéré
    init_weights = np.array([1/n_assets] * n_assets)
    
    # Maximiser le Sharpe = minimiser le Sharpe négatif
    result = minimize(
        lambda w: -calculate_sharpe(returns_df, w),
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x


def calculate_efficient_frontier(returns_df, n_points=20):
    """
    Calcule la frontière efficiente
    """
    n_assets = len(returns_df.columns)
    
    # Rendements cibles
    min_ret = returns_df.mean().min() * 252
    max_ret = returns_df.mean().max() * 252
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier = []
    
    for target in target_returns:
        try:
            weights = optimize_portfolio(returns_df, target_return=target)
            ret = calculate_portfolio_return(returns_df, weights)
            vol = calculate_portfolio_volatility(returns_df, weights)
            frontier.append({"return": ret, "volatility": vol, "weights": weights})
        except:
            continue
    
    return frontier
