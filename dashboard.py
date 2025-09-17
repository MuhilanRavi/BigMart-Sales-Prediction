import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒", layout="wide")
st.title("🛒 BigMart Sales Prediction Dashboard")

# Load data
try:
    results = pd.read_csv('results.csv')
    features = pd.read_csv('feature_importance.csv')
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(results['Actual_Sales'], results['Predicted_Sales']))
    mae = mean_absolute_error(results['Actual_Sales'], results['Predicted_Sales'])
    r2 = r2_score(results['Actual_Sales'], results['Predicted_Sales'])
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"{rmse:.2f}")
    with col2:
        st.metric("MAE", f"{mae:.2f}")
    with col3:
        st.metric("R² Score", f"{r2:.3f}")
    
    st.divider()
    
    # Main content
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Actual vs Predicted plot
        st.subheader("📊 Actual vs Predicted Sales")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.scatter(results['Actual_Sales'], results['Predicted_Sales'], alpha=0.6)
        
        # Perfect prediction line
        min_val = min(results['Actual_Sales'].min(), results['Predicted_Sales'].min())
        max_val = max(results['Actual_Sales'].max(), results['Predicted_Sales'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Sales')
        ax1.set_ylabel('Predicted Sales')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
    
    with right_col:
        # Feature importance
        st.subheader("🎯 Feature Importance")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = ax2.barh(features['Feature'], features['Importance'])
        
        # Color the bars
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FCE38A', '#F38181', '#A8E6CF']
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])
        
        ax2.set_xlabel('Importance')
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Feature importance table
        st.dataframe(features, use_container_width=True)
    
    # Sample predictions
    st.subheader("📋 Sample Predictions")
    st.dataframe(results.head(10), use_container_width=True)
    
except FileNotFoundError:
    st.error("Please make sure results.csv and feature_importance.csv are in the same folder!")
