import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Sepsis_final import *  # This imports function to our notebook

# --- Page Configuration ---
st.set_page_config(page_title="Sepsis Prediction Dashboard", page_icon="⚕️", layout="wide")

# --- Sidebar ---
st.sidebar.title("🔍 Sepsis Prediction App")
st.sidebar.subheader("Navigate")

page = st.sidebar.radio("Go to", ["Home", "Prediction", "Visualizations", "About"])

# --- Home Page ---
if page == "Home":
    st.title("Sepsis Prediction Dashboard")
    st.image("https://cdn.pixabay.com/photo/2017/08/30/07/54/heart-2698050_960_720.jpg", use_column_width=True)
    st.markdown("""
    ### 🎯 **Goal**  
    This app helps predict sepsis early based on patient data.
    
    ### ⚡ **Features**  
    - Make live predictions  
    - Visualize trends  
    - Explore patient stats  

    **Start by selecting a page from the sidebar ➡️**
    """)

# --- Prediction Page ---
if page == "Prediction":
    st.header("🩺 Make a Sepsis Prediction")
    st.write("Fill in the details below to get a prediction:")

    # User Input Form
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=80)
    with col2:
        temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
        resp_rate = st.number_input("Respiratory Rate", min_value=5, max_value=50, value=20)

    # Predict Button
    if st.button("Predict"):
        result = xgb_model.predict([[age, heart_rate, temp, resp_rate]])  # Modify based on your notebook's model
        st.success("✅ Sepsis Detected!" if result == 1 else "🟢 No Sepsis Detected")

# --- Visualization Page ---
if page == "Visualizations":
    st.header("📊 Data Visualizations")
    st.write("Explore patient data through charts and graphs.")

    # Example: Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- About Page ---
if page == "About":
    st.title("📘 About This Project")
    st.markdown("""
    - Built with **Streamlit**
    - Data sourced from your dataset
    - Model trained using the latest machine learning techniques
    
    💡 **Idea**: This app helps hospitals monitor and predict sepsis to save lives faster.
    """)

st.sidebar.info("🚀 Created by Your Name")
