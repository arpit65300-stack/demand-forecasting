import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Forecasting", page_icon="📦", layout="wide")
st.title("📦 E-Commerce Demand Forecasting")
st.markdown("Weekly demand forecasts for Olist product categories using XGBoost")

API_URL = "https://demand-forecasting-9sgs.onrender.com"

st.sidebar.header("Forecast Settings")
category = st.sidebar.selectbox(
    "Product Category",
    ["cama_mesa_banho", "beleza_saude"],
    format_func=lambda x: "Bed/Bath/Table" if x == "cama_mesa_banho" else "Health & Beauty"
)
weeks_ahead = st.sidebar.slider("Weeks to Forecast", min_value=1, max_value=12, value=4)

if st.sidebar.button("Generate Forecast", type="primary"):
    with st.spinner("Fetching forecast..."):
        try:
            response = requests.post(
                f"{API_URL}/forecast",
                json={"category": category, "weeks_ahead": weeks_ahead}
            )
            data = response.json()
            forecast_df = pd.DataFrame(data["forecast"])
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Category", "Bed/Bath/Table" if category == "cama_mesa_banho" else "Health & Beauty")
            col2.metric("Weeks Forecast", weeks_ahead)
            col3.metric("Avg Predicted Units", f"{forecast_df['predicted_units'].mean():.0f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["predicted_units"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#667eea", width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title=f"Weekly Demand Forecast — {weeks_ahead} weeks ahead",
                xaxis_title="Date",
                yaxis_title="Predicted Units Sold",
                hovermode="x unified",
                plot_bgcolor="white",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast Table")
            forecast_df.columns = ["Date", "Predicted Units"]
            st.dataframe(forecast_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error connecting to API: {e}")
else:
    st.info("Select a category and click Generate Forecast to get started")

st.markdown("---")
st.markdown("Built with XGBoost + FastAPI + Streamlit | Olist Brazilian E-Commerce Dataset")
