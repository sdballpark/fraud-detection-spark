import streamlit as st
import pandas as pd

st.title('🚨 Fraud Detection - PySpark Model Demo')

uploaded_file = st.file_uploader("Upload a CSV with transactions", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write('Sample Uploaded Data:', data.head())

    st.success('⚡ This is a demo placeholder! Model prediction to be integrated.')
