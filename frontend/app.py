import streamlit as st
import requests
import pandas as pd

st.title("AI Sustainability Advisor")

uploaded = st.file_uploader("Upload Waste Image")

if uploaded:
    files = {"file": uploaded}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    result = response.json()

    st.success(f"Predicted: {result['class']}")

# Dashboard
st.subheader("Live Waste Statistics")

try:
    df = pd.read_csv("dashboard_output.csv")
    st.bar_chart(df.set_index("waste_type"))
except:
    st.write("No data yet")