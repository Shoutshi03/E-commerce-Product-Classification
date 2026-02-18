import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIG

st.set_page_config(
    page_title="NLP Product Classifier Dashboard",
    layout="wide"
)

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/ecommerceDataset.csv"

PRIMARY_COLOR = "#3B82F6"
SECONDARY_COLOR = "#6366F1"


# CUSTOM CSS

st.markdown(f"""
<style>

.main {{
    background-color: #F8FAFC;
}}

/* Streamlit button */
div.stButton > button {{
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
    border: none;
}}

/* Hover effect */
div.stButton > button:hover {{
    background-color: #1E40AF !important;
    color: white !important;
}}


.form-card {{
    background-color: #F1F5F9;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #E2E8F0;
}}

.form-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}}

.load-example {{
    color: #2563EB;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
}}

input, textarea {{
    border-radius: 10px !important;
}}

.predict-btn {{
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    color: white;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
}}


.card {{
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}

.prediction-card {{
    background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
    padding: 25px;
    border-radius: 14px;
    color: white;
}}

.tag {{
    display: inline-block;
    background-color: #EEF2FF;
    color: #3730A3;
    padding: 6px 12px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 6px;
    font-size: 13px;
}}

.big-text {{
    font-size: 34px;
    font-weight: bold;
}}

.small-label {{
    font-size: 13px;
    opacity: 0.8;
}}

.reason-box {{
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}}

</style>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()


# HEADER

st.markdown("""
# Ecommerce Product Classification
TF-IDF Vectorization & Multi-class Classification
""")

tabs = st.tabs(["Live Classification"])


# TAB 0 — LIVE CLASSIFICATION

with tabs[0]:

    col1, col2 = st.columns([1, 1.2])

    # ================= LEFT SIDE FORM =================
    with col1:
        st.markdown("## Ecommerce classification product")
        st.markdown("""
                    this is an 
                    E-commerce Product Classification program that classifies products from its title and 
                    a short description of the product.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        # Header with Load Example
        st.markdown("""
        <div class="form-header">
            <h3>Product Details</h3>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Load Example"):
            st.session_state["title"] = "Samsung Galaxy S23 Ultra"
            st.session_state["description"] = "Latest Samsung flagship smartphone with Snapdragon processor and 200MP camera."

        # INPUT WITH PLACEHOLDER (disappears when typing automatically)
        title = st.text_input(
            "Product Title",
            value=st.session_state.get("title", ""),
            placeholder="e.g. Samsung Galaxy S23 Ultra"
        )

        description = st.text_area(
            "Product Description",
            value=st.session_state.get("description", ""),
            placeholder="Enter a detailed description of the product...",
            height=150
        )

        predict_clicked = st.button("Predict Category", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ================= RIGHT SIDE RESULT =================
    with col2:

        if predict_clicked:

            text = title + " " + description
            prediction = model.predict([text])[0]

            if hasattr(model.named_steps["clf"], "predict_proba"):
                confidence = np.max(model.predict_proba([text])) * 100
            else:
                confidence = 97.0

            st.markdown(f"""
            <div class="prediction-card">
                <h3>Prediction Result</h3>
                <p class="small-label">Predicted Category</p>
                <p class="big-text">{prediction}</p>
                <p class="small-label">Confidence Score</p>
                <p class="big-text">{confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

