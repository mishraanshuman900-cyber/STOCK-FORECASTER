import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from datetime import datetime, timedelta

# Hide statistical warnings 
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Institutional Stock Forecaster", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM DARK MODE CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    div.stButton > button:first-child {
        background-color: #00f2fe;
        color: #000000;
        font-weight: bold;
        border-radius: 6px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #4facfe;
        color: #ffffff;
    }
    [data-testid="stMetricValue"] {
        color: #00f2fe;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Model Configuration")

# Timeline Selection - Expanded to capture up to 20 years of historical cycles
duration_years = st.sidebar.slider("Historical Baseline Range (Years)", min_value=5, max_value=20, value=10)
end_date = datetime.today()
start_date = end_date - timedelta(days=duration_years * 365)

# Forecasting Macro Fine-Tuning 
st.sidebar.markdown("### Fine-Tuning")
forecast_years = st.sidebar.slider("Forecast Horizon (Years)", min_value=1, max_value=5, value=5)

# Convert chosen calendar years to structural business trading days (252 days/year)
forecast_days = int(forecast_years * 252)

trend_type = st.sidebar.selectbox("Trend Component", ["additive", "damped", "none"], index=0)
use_boxcox = st.sidebar.checkbox("Apply Box-Cox Transform (Stabilizes long-term variance)", value=True)

# --- MAIN INTERFACE ---
st.title("🏛️ Professional Stock Analyzer & Forecaster (5-Year Outlook)")
st.markdown("Search global equities using Yahoo Finance tickers to analyze price distribution and generate macro-level time-series projections.")

# Search Bar for Stock Ticker
ticker_input = st.text_input("🔍 Enter Stock Ticker Symbol (e.g., AAPL, MSFT, NVDA, TSLA):", value="AAPL").strip().upper()

if ticker_input:
    try:
        with st.spinner(f"Fetching structural market data for {ticker_input}..."):
            stock = yf.Ticker(ticker_input)
            df = stock.history(start=start_date, end=end_date)
            info = stock.info
            
        if df.empty:
            st.error(f"No data found for ticker '{ticker_input}'. Please check the symbol and try again.")
    except Exception as e:
        st.error(f"Error accessing Yahoo Finance API: {e}")
        df = pd.DataFrame()

    if not df.empty:
        company_name = info.get('longName', ticker_input)
        currency = info.get('currency', 'USD')
        st.success(f"Connected to live feed for **{company_name}** ({ticker_input})")
        
        # Extract closing values
        data_series = df['Close'].dropna()
        latest_price = data_series.iloc[-1]
        
        # --- STATISTICAL ANALYSIS ---
        mean_val = data_series.mean()
        median_val = data_series.median()
        
        threshold = 0.05 * mean_val
        is_symmetric = abs(mean_val - median_val) <= threshold

        # Summary Display Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Latest Closing Price", value=f"{latest_price:,.2f} {currency}")
        with m2:
            st.metric(label="Historical Mean", value=f"{mean_val:,.2f}")
        with m3:
            st.metric(label="Historical Median", value=f"{median_val:,.2f}")

        if is_symmetric:
            st.info("ℹ️ **Distribution Assessment:** The historical mean and median are closely aligned, showing balanced price consolidation zones.")
        else:
            st.warning("ℹ️ **Distribution Assessment:** The mean and median deviate, signifying a strong long-term structural trend or skew in historical pricing.")

        # --- VISUALIZATIONS TABS ---
        tab1, tab2 = st.tabs(["📉 Price Forecast Engine", "📊 Distribution & Spread"])

        with tab1:
            st.subheader(f"{forecast_years}-Year Predictive Analytics Horizon")
            
            with st.spinner("Optimizing algorithmic parameters for multi-year projection..."):
                try:
                    if trend_type == "damped":
                        fit_trend = "add"
                        damped_setting = True
                    elif trend_type == "additive":
                        fit_trend = "add"
                        damped_setting = False
                    else:
                        fit_trend = None
                        damped_setting = False

                    # Fit Holt-Winters model optimized for macro trends
                    model = ExponentialSmoothing(
                        data_series.values, 
                        trend=fit_trend, 
                        damped_trend=damped_setting,
                        seasonal=None,
                        use_boxcox=True if use_boxcox else False
                    )
                    fit_model = model.fit(optimized=True)
                    forecast_values = fit_model.forecast(forecast_days)
                    
                    # Generate multi-year dates mapping back to real business days
                    last_date = data_series.index[-1]
                    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')

                    # Build High-Fidelity Chart
                    fig = go.Figure()
                    
                    # Historical Data (Neon Cyan)
                    fig.add_trace(go.Scatter(
                        x=data_series.index, y=data_series.values,
                        mode='lines', name='Historical Close',
                        line=dict(color='#00f2fe', width=2)
                    ))
                    
                    # Macro Forecast (Neon Magenta)
                    fig.add_trace(go.Scatter(
                        x=forecast_dates, y=forecast_values,
                        mode='lines', name=f'{forecast_years}-Year Algorithmic Projection',
                        line=dict(color='#fe0979', width=3, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"{company_name} ({ticker_input}) Multi-Year Forward Horizon Projection",
                        xaxis_title="Timeline",
                        yaxis_title=f"Price ({currency})",
                        hovermode="x unified",
                        template="plotly_dark",
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="left", 
                            x=0
                        ),
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as model_err:
                    st.error(f"Mathematical engine encountered an optimization failure: {model_err}")

        with tab2:
            st.subheader("Asset Volatility & Density Spread")
            
            fig_hist = px.histogram(
                df, x="Close", 
                nbins=40,
                title=f"Density Breakdown for {ticker_input}",
                color_discrete_sequence=['#00e676'],
                marginal="box",
                labels={"Close": f"Closing Price ({currency})"}
            )
            fig_hist.update_layout(
                template="plotly_dark", 
                bargap=0.05,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
