"""
Chargement des données depuis Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# Actifs par défaut
DEFAULT_ASSETS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SPY": "S&P 500 ETF",
    "GLD": "Or",
    "TLT": "Obligations US"
}


def load_asset_data(symbol, start_date, end_date):
    """
    Charge les données d'un actif
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if not data.empty:
            return data['Close']
        return None
    except Exception as e:
        print(f"❌ Erreur pour {symbol}: {e}")
        return None


def load_portfolio_data(symbols, period="2y"):
    """
    Charge les données de plusieurs actifs
    """
    end_date = datetime.now()
    
    if period == "1y":
        start_date = end_date - timedelta(days=365)
    elif period == "2y":
        start_date = end_date - timedelta(days=730)
    elif period == "5y":
        start_date = end_date - timedelta(days=1825)
    else:
        start_date = end_date - timedelta(days=730)
    
    data = {}
    for symbol in symbols:
        prices = load_asset_data(symbol, start_date, end_date)
        if prices is not None and len(prices) > 20:
            data[symbol] = prices
            print(f"✅ {symbol}: {len(prices)} points")
    
    if not data:
        return None
    
    df = pd.DataFrame(data).dropna()
    return df
