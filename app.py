import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# -------------------------------
# Load trained model
# -------------------------------
model = load("./saved_model/random_forest.joblib")

# -------------------------------
# Load dataset to get symptom list
# -------------------------------
df = pd.read_csv("./dataset/training_data.csv")

# first 132 columns are symptoms
symptoms = list(df.columns[:-2])

# -------------------------------
# Page title
# -------------------------------
st.title("🩺 Disease Prediction System")
st.write("Select symptoms to predict possible diseases")

# -------------------------------
# Symptom selection
# -------------------------------
selected_symptoms = st.multiselect(
    "Choose Symptoms",
    symptoms
)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")
    else:

        # create input vector
        input_data = [0] * len(symptoms)

        for symptom in selected_symptoms:
            index = symptoms.index(symptom)
            input_data[index] = 1

        # convert to dataframe
        input_df = pd.DataFrame([input_data], columns=symptoms)

        # get probabilities
        probs = model.predict_proba(input_df)[0]
        diseases = model.classes_

        # get top 3 predictions
        top3 = np.argsort(probs)[-3:][::-1]

        st.subheader("Top 3 Possible Diseases")

        for i in top3:
            st.write(f"**{diseases[i]} — {probs[i]*100:.2f}%**")

        # optional chart
        st.subheader("Prediction Confidence")

        chart_data = {
            diseases[i]: probs[i]*100
            for i in top3
        }

        st.bar_chart(chart_data)