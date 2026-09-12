"""
Télécharge et stocke les données localement
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def download_crypto_coingecko(coin_id, days=365):
    """
    Télécharge l'historique depuis CoinGecko (gratuit, sans clé)
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date")[["price"]]
    df.columns = ["Close"]
    
    return df


def download_crypto_coinpaprika(coin_id, days=365):
    """
    Télécharge l'historique depuis CoinPaprika (gratuit, sans clé)
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}/historical"
    params = {"start": start_date, "interval": "1d"}
    
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.set_index("date")[["price"]]
    df.columns = ["Close"]
    
    return df


def download_brvm(ticker, period="daily"):
    """
    Télécharge l'historique BRVM depuis le dépôt GitHub
    """
    url = f"https://raw.githubusercontent.com/Fredysessie/brvm-data-public/main/data/{ticker}/{ticker}.{period}.csv"
    
    try:
        df = pd.read_csv(url)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return df
    except Exception as e:
        print(f"❌ Erreur BRVM {ticker}: {e}")
        return None


def download_all_defaults():
    """
    Télécharge tous les actifs par défaut
    """
    assets = {
        "BTC": ("bitcoin", "BTC-USD"),
        "ETH": ("ethereum", "ETH-USD"),
        "SPY": ("spy", "SPY"),
        "GLD": ("gold", "GLD"),
        "TLT": ("tlt", "TLT")
    }
    
    for symbol, (cg_id, filename) in assets.items():
        print(f"📥 Téléchargement {symbol}...")
        
        # Essayer CoinPaprika d'abord
        df = download_crypto_coinpaprika(cg_id)
        
        if df is not None and not df.empty:
            df.to_csv(f"{DATA_DIR}/{filename}.csv")
            print(f"✅ {symbol}: {len(df)} points")
        else:
            print(f"❌ {symbol}: échec")


if __name__ == "__main__":
    print("📥 Téléchargement des données...")
    download_all_defaults()
    print("✅ Terminé")
