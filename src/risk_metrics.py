"""
Métriques de risque pour portefeuille
VaR, Expected Shortfall, volatilité, drawdown
"""
import numpy as np
import pandas as pd
from scipy import stats


def calculate_returns(prices):
    """
    Calcule les rendements logarithmiques
    """
    if isinstance(prices, pd.Series):
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = np.diff(np.log(prices))
        returns = returns[~np.isnan(returns)]
    return returns


def calculate_volatility(returns, annualize=True):
    """
    Calcule la volatilité (écart-type des rendements)
    """
    vol = np.std(returns)
    if annualize:
        vol = vol * np.sqrt(252)  # 252 jours de trading par an
    return vol


def calculate_var_historical(returns, confidence=0.95):
    """
    VaR historique (percentile des pertes)
    """
    var = np.percentile(returns, (1 - confidence) * 100)
    return var


def calculate_var_parametric(returns, confidence=0.95):
    """
    VaR paramétrique (loi normale)
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    z = stats.norm.ppf(1 - confidence)
    var = mu + z * sigma
    return var


def calculate_var_cornish_fisher(returns, confidence=0.95):
    """
    VaR avec expansion de Cornish-Fisher (tient compte de skewness et kurtosis)
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    z = stats.norm.ppf(1 - confidence)
    
    # Skewness et kurtosis
    s = stats.skew(returns)
    k = stats.kurtosis(returns)
    
    # Expansion de Cornish-Fisher
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24 - (2*z**3 - 5*z) * s**2 / 36
    
    var = mu + z_cf * sigma
    return var


def calculate_expected_shortfall(returns, confidence=0.95):
    """
    Expected Shortfall (perte moyenne au-delà de la VaR)
    """
    var = calculate_var_historical(returns, confidence)
    es = returns[returns <= var].mean()
    return es


def calculate_max_drawdown(prices):
    """
    Drawdown maximal
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    cumulative = np.cumprod(1 + calculate_returns(prices))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = np.min(drawdown)
    return max_dd


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Ratio de Sharpe (rendement ajusté au risque)
    """
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """
    Ratio de Sortino (ne pénalise que la volatilité à la baisse)
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns)
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
    return sortino


def calculate_all_metrics(returns, prices):
    """
    Calcule toutes les métriques de risque
    """
    return {
        "volatilite_annuelle": calculate_volatility(returns),
        "var_95_historique": calculate_var_historical(returns, 0.95),
        "var_99_historique": calculate_var_historical(returns, 0.99),
        "var_95_parametrique": calculate_var_parametric(returns, 0.95),
        "var_99_parametrique": calculate_var_parametric(returns, 0.99),
        "var_95_cornish_fisher": calculate_var_cornish_fisher(returns, 0.95),
        "es_95": calculate_expected_shortfall(returns, 0.95),
        "es_99": calculate_expected_shortfall(returns, 0.99),
        "max_drawdown": calculate_max_drawdown(prices),
        "sharpe": calculate_sharpe_ratio(returns),
        "sortino": calculate_sortino_ratio(returns),
        "skewness": stats.skew(returns),
        "kurtosis": stats.kurtosis(returns),
    }
