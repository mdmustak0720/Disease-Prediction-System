import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .disease-high {
        color: #d32f2f;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .disease-medium {
        color: #f57c00;
        font-size: 1.1rem;
    }
    .disease-low {
        color: #388e3c;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    return load("./saved_model/random_forest.joblib")

# -------------------------------
# Load dataset to get symptom list
# -------------------------------
@st.cache_data
def load_symptoms():
    df = pd.read_csv("./dataset/training_data.csv")
    return list(df.columns[:-2])

model = load_model()
symptoms = load_symptoms()

# -------------------------------
# Page title and description
# -------------------------------
st.markdown("# 🩺 Disease Prediction System")
st.markdown("Select your symptoms to get predictions for possible diseases")
st.markdown("---")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Select Symptoms")
    selected_symptoms = st.multiselect(
        "Choose one or more symptoms from the list:",
        symptoms,
        key="symptoms_select"
    )
    
    symptom_count = len(selected_symptoms)
    st.info(f"✅ {symptom_count} symptom(s) selected")

with col2:
    st.subheader("ℹ️ Instructions")
    st.markdown("""
    1. Select symptoms from the list
    2. Click the "Predict Disease" button
    3. View predictions and confidence scores
    """)

st.markdown("---")

# Store prediction results in session state to prevent re-rendering
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# Prediction button
predict_col1, predict_col2, predict_col3 = st.columns([2, 1, 2])

with predict_col2:
    if st.button("🔍 Predict Disease", use_container_width=True):
        if len(selected_symptoms) == 0:
            st.error("❌ Please select at least one symptom")
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
            top3_indices = np.argsort(probs)[-3:][::-1]

            # Store in session state
            st.session_state.prediction_data = {
                'diseases': diseases,
                'probs': probs,
                'top3_indices': top3_indices
            }

# Display predictions if available
if st.session_state.prediction_data is not None:
    data = st.session_state.prediction_data
    diseases = data['diseases']
    probs = data['probs']
    top3_indices = data['top3_indices']

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    # Create results columns
    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        st.markdown("### Top 3 Possible Diseases")
        
        for idx, disease_idx in enumerate(top3_indices, 1):
            disease_name = diseases[disease_idx]
            confidence = probs[disease_idx] * 100
            
            # Color code based on confidence
            if confidence >= 50:
                color_class = "disease-high"
            elif confidence >= 30:
                color_class = "disease-medium"
            else:
                color_class = "disease-low"
            
            st.markdown(f"""
            <div class="prediction-box">
                <strong>#{idx}</strong> <span class="{color_class}">{disease_name}</span>
                <br>Confidence: <strong>{confidence:.2f}%</strong>
            </div>
            """, unsafe_allow_html=True)

    with result_col2:
        st.markdown("### Confidence Chart")
        
        # Create a stable matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        chart_diseases = [diseases[i] for i in top3_indices]
        chart_probs = [probs[i] * 100 for i in top3_indices]
        
        # Color bars based on confidence
        colors = ['#d32f2f' if p >= 50 else '#f57c00' if p >= 30 else '#388e3c' 
                  for p in chart_probs]
        
        bars = ax.barh(chart_diseases, chart_probs, color=colors)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, chart_probs)):
            ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold')
        
        ax.set_xlabel('Confidence (%)', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)

    # Additional metrics
    st.markdown("---")
    st.subheader("📊 All Predictions")
    
    # Create dataframe with all predictions
    all_predictions = pd.DataFrame({
        'Disease': diseases,
        'Confidence (%)': np.round(probs * 100, 2)
    }).sort_values('Confidence (%)', ascending=False)
    
    st.dataframe(all_predictions, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <small>⚠️ Disclaimer: This system is for informational purposes only and should not replace 
    professional medical diagnosis. Please consult a healthcare professional for accurate diagnosis.</small>
</div>
""", unsafe_allow_html=True)