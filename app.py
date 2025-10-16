import streamlit as st
import joblib
import pandas as pd
import numpy as np

model_rf = joblib.load('rf_model.pkl')
label_enc = joblib.load('label_encoder.pkl')

st.title("Formula 1 DNF Classification")
st.write('Predict Did not finish [DNF] using the race details')

year = st.number_input("Year", min_value=1950, max_value=2025)
grid = st.number_input("Starting Grid Position", min_value=1, max_value=30)
laps = st.number_input("Laps Completed", min_value=0, max_value=100)
points = st.number_input("Driver Points", min_value=0.0)
fastestLapTime = st.number_input("Fastest Lap Time (seconds)", min_value=0.0)
rank = st.number_input("Driver Rank", min_value=1, max_value=100)
country = st.text_input("Country", "UK")

input_data = pd.DataFrame({
    "year": [year],
    "grid": [grid],
    "laps": [laps],
    "points": [points],
    "fastestLapTime": [fastestLapTime],
    "rank": [rank],
    "country": [country.lower()],
})


input_data["country"] = label_enc.fit_transform(input_data["country"])

if st.button("Predict"):
    prediction = model_rf.predict(input_data.reindex(columns=model_rf.feature_names_in_, fill_value=0))
    if prediction == 1:
        st.success("Predicted DNF: YES")
    else:
        st.success("Predicted DNF: NO")
