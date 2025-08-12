"""
Streamlit dashboard for the AI-Powered Underwriting Engine.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Page configuration
st.set_page_config(
    page_title="AI-Powered Underwriting Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main dashboard application."""
    st.title("🏦 AI-Powered Underwriting Engine")
    st.markdown("Real-time credit risk assessment for small business financing")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Risk Assessment", "Portfolio Overview", "Model Performance", "Data Sources"]
    )
    
    if page == "Risk Assessment":
        risk_assessment_page()
    elif page == "Portfolio Overview":
        portfolio_overview_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Data Sources":
        data_sources_page()

def risk_assessment_page():
    """Risk assessment interface."""
    st.header("Business Risk Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Business Information")
        business_id = st.text_input("Business ID")
        business_name = st.text_input("Business Name")
        industry = st.selectbox("Industry", [
            "Food Service", "Retail", "Technology", "Healthcare",
            "Manufacturing", "Professional Services", "Other"
        ])
        location = st.text_input("Location")
        years_in_business = st.number_input("Years in Business", min_value=0, max_value=100, value=3)
        annual_revenue = st.number_input("Annual Revenue ($)", min_value=0, value=500000)
        
        if st.button("Assess Risk"):
            if business_id and business_name:
                # TODO: Make actual API call to assessment endpoint
                risk_score = 0.75
                risk_category = "Medium Risk"
                
                st.success(f"Risk assessment completed for {business_name}")
                st.metric("Risk Score", f"{risk_score:.2f}")
                st.metric("Risk Category", risk_category)
            else:
                st.error("Please fill in Business ID and Name")
    
    with col2:
        st.subheader("Risk Profile Visualization")
        
        # Sample risk factors data
        risk_factors = {
            "Factor": ["Transaction Consistency", "Customer Sentiment", "Business Age", 
                      "Review Volume", "Geographic Risk"],
            "Importance": [0.23, 0.18, 0.15, 0.12, 0.10],
            "Score": [0.8, 0.6, 0.9, 0.7, 0.5]
        }
        
        df_factors = pd.DataFrame(risk_factors)
        
        # Feature importance chart
        fig_importance = px.bar(
            df_factors, x="Importance", y="Factor", 
            orientation="h", title="Key Risk Factors (Importance)"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

def portfolio_overview_page():
    """Portfolio overview dashboard."""
    st.header("Portfolio Overview")
    
    # Sample portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", "1,234", "12%")
    with col2:
        st.metric("Approved", "987", "8%")
    with col3:
        st.metric("Default Rate", "3.2%", "-0.5%")
    with col4:
        st.metric("Portfolio Value", "$45.6M", "15%")
    
    # Sample data for charts
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="M")
    applications = [100, 120, 110, 150, 180, 200, 220, 190, 210, 240, 250, 280]
    
    df_timeline = pd.DataFrame({"Date": dates, "Applications": applications})
    
    fig_timeline = px.line(df_timeline, x="Date", y="Applications", 
                          title="Monthly Application Volume")
    st.plotly_chart(fig_timeline, use_container_width=True)

def model_performance_page():
    """Model performance metrics."""
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        metrics_data = {
            "Metric": ["AUC-ROC", "Precision", "Recall", "F1-Score"],
            "Value": [0.87, 0.83, 0.79, 0.81]
        }
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
    
    with col2:
        st.subheader("Model Drift Monitoring")
        # Sample drift data
        drift_dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="W")
        drift_scores = [0.02, 0.03, 0.015, 0.025, 0.04, 0.02, 0.035] * 8
        
        df_drift = pd.DataFrame({"Date": drift_dates[:len(drift_scores)], "Drift_Score": drift_scores})
        
        fig_drift = px.line(df_drift, x="Date", y="Drift_Score", 
                           title="Model Drift Over Time")
        st.plotly_chart(fig_drift, use_container_width=True)

def data_sources_page():
    """Data sources status."""
    st.header("Data Sources Status")
    
    # Sample data source status
    sources_data = {
        "Source": ["Yelp API", "Google Maps", "Twitter", "Shopify", "Internal DB"],
        "Status": ["Active", "Active", "Active", "Inactive", "Active"],
        "Last_Update": ["2 min ago", "5 min ago", "1 min ago", "N/A", "30 sec ago"],
        "Records_Today": [1250, 890, 2100, 0, 5600]
    }
    
    df_sources = pd.DataFrame(sources_data)
    
    # Color code status
    def color_status(val):
        color = 'green' if val == 'Active' else 'red'
        return f'background-color: {color}; color: white'
    
    styled_df = df_sources.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)

if __name__ == "__main__":
    main()