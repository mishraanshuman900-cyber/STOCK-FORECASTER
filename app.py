import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Ignore statistical warnings for a cleaner UI
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Data Analyzer & Forecaster", page_icon="📈", layout="wide")

# --- CUSTOM CSS FOR A PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Advanced Data Analyzer & Forecaster")
st.markdown("Upload your Excel file to automatically analyze numerical distributions and generate dynamic forecasts.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx, .xls)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head())

        # Select only numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.error("No numerical columns found in the dataset.")
        else:
            st.markdown("---")
            st.header("📈 Column Analysis & Forecasting")
            
            # Create a dropdown to select which column to analyze
            target_col = st.selectbox("Select a numerical column to analyze:", numeric_cols)
            
            # --- CALCULATE MEAN & MEDIAN ---
            col_data = df[target_col].dropna()
            mean_val = col_data.mean()
            median_val = col_data.median()
            
            # Define "close" as being within 5% of each other
            threshold = 0.05 * mean_val if mean_val != 0 else 0
            is_close = abs(mean_val - median_val) <= threshold

            # --- DISPLAY METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Mean", value=f"{mean_val:,.2f}")
            col2.metric(label="Median", value=f"{median_val:,.2f}")
            
            with col3:
                if is_close:
                    st.success("✅ Mean & Median are close. Data is highly symmetrical and optimal for modeling.")
                else:
                    st.warning("⚠️ Mean & Median are significantly different. Data is skewed.")

            st.markdown("---")
            
            # --- INTERACTIVE HISTOGRAM ---
            st.subheader(f"Distribution of {target_col}")
            fig_hist = px.histogram(
                df, x=target_col, 
                nbins=30, 
                title=f"Histogram of {target_col}",
                color_discrete_sequence=['#3498db'],
                marginal="box" # Adds a professional box plot at the top
            )
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- DYNAMIC FORECASTING ---
            st.markdown("---")
            st.subheader(f"🚀 Dynamic Forecast for {target_col}")
            st.info("Assuming the data points are chronologically ordered, here is a forward projection using Holt-Winters Exponential Smoothing.")
            
            # Prepare data for forecasting
            ts_data = col_data.values
            forecast_steps = st.slider("Select number of periods to forecast forward:", min_value=5, max_value=50, value=12)
            
            try:
                # Apply Holt-Winters Model (Trend enabled to avoid "flat" lines)
                # Fallback to just trend if data isn't long enough for seasonality
                model = ExponentialSmoothing(ts_data, trend='add', seasonal=None, initialization_method="estimated")
                fit_model = model.fit()
                forecast = fit_model.forecast(forecast_steps)
                
                # Create historical and forecast indices
                history_x = list(range(len(ts_data)))
                forecast_x = list(range(len(ts_data), len(ts_data) + forecast_steps))
                
                # Plotly Professional Line Chart
                fig_forecast = go.Figure()
                
                # Historical Data Line
                fig_forecast.add_trace(go.Scatter(
                    x=history_x, y=ts_data, 
                    mode='lines+markers', 
                    name='Historical Data',
                    line=dict(color='#2c3e50', width=2)
                ))
                
                # Forecast Data Line
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_x, y=forecast, 
                    mode='lines+markers', 
                    name='Forecast Prediction',
                    line=dict(color='#e74c3c', width=3, dash='dash')
                ))
                
                fig_forecast.update_layout(
                    title=f"Trend Analysis & Forecast ({forecast_steps} periods)",
                    xaxis_title="Time / Index",
                    yaxis_title="Value",
                    hovermode="x unified",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
            except Exception as e:
                st.error(f"Not enough variation or data points to generate a dynamic forecast. Ensure your data has a clear trend. Error details: {e}")

    except Exception as e:
        st.error(f"Error reading file: {e}")
