"""
Portfolio Risk Dashboard
Analyse de risque pour portefeuille multi-actifs
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_portfolio_data, DEFAULT_ASSETS
from src.risk_metrics import calculate_returns, calculate_all_metrics
from src.portfolio import optimize_portfolio, calculate_efficient_frontier

# Configuration
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .stApp { background-color: #0a0a1a; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .label { color: #8892b0; font-size: 12px; }
    .metric-card .value { color: #ffffff; font-size: 24px; font-weight: 700; }
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# Titre
st.title("📊 Portfolio Risk Dashboard")
st.markdown("Analyse de risque pour portefeuille multi-actifs")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sélection des actifs
    all_symbols = list(DEFAULT_ASSETS.keys())
    selected = st.multiselect(
        "Actifs du portefeuille",
        all_symbols,
        default=["BTC-USD", "ETH-USD", "SPY", "GLD"]
    )
    
    period = st.selectbox("Période", ["1y", "2y", "5y"], index=1)
    
    if st.button("🔄 Analyser", type="primary"):
        st.rerun()

# Chargement
if not selected:
    st.warning("Sélectionnez au moins un actif")
    st.stop()

with st.spinner("Chargement des données..."):
    prices_df = load_portfolio_data(selected, period)

if prices_df is None or prices_df.empty:
    st.error("❌ Impossible de charger les données")
    st.stop()

# Calculs
returns_df = prices_df.pct_change().dropna()
n_assets = len(selected)
weights_eq = np.array([1/n_assets] * n_assets)

# Métriques du portefeuille équipondéré
portfolio_returns = returns_df.dot(weights_eq)
portfolio_prices = (1 + portfolio_returns).cumprod() * 100

metrics = calculate_all_metrics(portfolio_returns, portfolio_prices)

# Métriques principales
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Rendement annualisé</div>
        <div class='value'>{portfolio_returns.mean() * 252 * 100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Volatilité</div>
        <div class='value'>{metrics['volatilite_annuelle'] * 100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>VaR 95%</div>
        <div class='value' style='color: #FFB74D;'>{metrics['var_95_historique'] * 100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>VaR 99%</div>
        <div class='value' style='color: #FF4444;'>{metrics['var_99_historique'] * 100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Max Drawdown</div>
        <div class='value' style='color: #FF4444;'>{metrics['max_drawdown'] * 100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Graphiques
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Évolution du portefeuille")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_prices.index,
        y=portfolio_prices.values,
        mode='lines',
        name='Portefeuille',
        line=dict(color="#6C63FF", width=2)
    ))
    fig.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Distribution des rendements")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=portfolio_returns,
        nbinsx=50,
        name='Rendements',
        marker_color="#6C63FF"
    ))
    fig.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig, use_container_width=True)

# Allocation optimale
st.markdown("---")
st.subheader("🎯 Allocation optimale (Markowitz)")

try:
    optimal_weights = optimize_portfolio(returns_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=selected,
            values=optimal_weights,
            hole=0.4
        )])
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        opt_return = np.sum(returns_df.mean() * optimal_weights) * 252
        opt_vol = np.sqrt(optimal_weights.T @ (returns_df.cov() * 252) @ optimal_weights)
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='label'>Rendement optimal</div>
            <div class='value'>{opt_return * 100:.2f}%</div>
        </div>
        <br>
        <div class='metric-card'>
            <div class='label'>Volatilité optimale</div>
            <div class='value'>{opt_vol * 100:.2f}%</div>
        </div>
        <br>
        <div class='metric-card'>
            <div class='label'>Ratio de Sharpe</div>
            <div class='value'>{opt_return / opt_vol:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.warning(f"Optimisation impossible: {e}")

# Matrice de corrélation
st.markdown("---")
st.subheader("🔗 Matrice de corrélation")

corr_matrix = returns_df.corr()
fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1
)
fig.update_layout(template='plotly_dark', height=400)
st.plotly_chart(fig, use_container_width=True)

# Tableau détaillé
st.markdown("---")
st.subheader("📋 Métriques détaillées")

metrics_df = pd.DataFrame([
    {"Métrique": "Volatilité annualisée", "Valeur": f"{metrics['volatilite_annuelle']*100:.2f}%"},
    {"Métrique": "VaR 95% (historique)", "Valeur": f"{metrics['var_95_historique']*100:.2f}%"},
    {"Métrique": "VaR 99% (historique)", "Valeur": f"{metrics['var_99_historique']*100:.2f}%"},
    {"Métrique": "VaR 95% (Cornish-Fisher)", "Valeur": f"{metrics['var_95_cornish_fisher']*100:.2f}%"},
    {"Métrique": "Expected Shortfall 95%", "Valeur": f"{metrics['es_95']*100:.2f}%"},
    {"Métrique": "Expected Shortfall 99%", "Valeur": f"{metrics['es_99']*100:.2f}%"},
    {"Métrique": "Max Drawdown", "Valeur": f"{metrics['max_drawdown']*100:.2f}%"},
    {"Métrique": "Ratio de Sharpe", "Valeur": f"{metrics['sharpe']:.2f}"},
    {"Métrique": "Ratio de Sortino", "Valeur": f"{metrics['sortino']:.2f}"},
    {"Métrique": "Skewness", "Valeur": f"{metrics['skewness']:.3f}"},
    {"Métrique": "Kurtosis", "Valeur": f"{metrics['kurtosis']:.3f}"},
])

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #4a4a5e; font-size: 12px;'>
    Portfolio Risk Dashboard · {datetime.now().strftime('%d/%m/%Y')} · 
    ⚠️ Ceci n'est pas un conseil financier
</div>
""", unsafe_allow_html=True)
