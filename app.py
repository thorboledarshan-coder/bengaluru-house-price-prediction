from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import numpy as np
import pandas as pd
import pickle


@st.cache_resource
def load_model():
    with open("bangalore_house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Load model
model = load_model()

# Categorical and numeric columns
categorical_cols = ['location']
numeric_cols = ['total_sqft', 'bath', 'balcony', 'bhk']

# Get location options from encoder
ohe = model.named_steps['preprocessing'].named_transformers_['location_encoder']
location_categories = ohe.categories_[0].tolist()



def predict_price(location, total_sqft, bath, balcony, bhk):
    # Build a one-row DataFrame with the same columns the pipeline expects.
    sample_df = pd.DataFrame({
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "balcony": [balcony],
        "bhk": [bhk]
    })

    # Pass the raw DataFrame to the pipeline so its ColumnTransformer and encoders run internally.
    price_lakhs = model.predict(sample_df)[0]
    return price_lakhs



# STREAMLIT UI code


st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Bengaluru House Price Prediction")
st.write(
    "This app predicts the **price of a house in Bengaluru** based on location, "
    "square footage, number of bathrooms, balconies, and BHK."
)

st.markdown("---")

# Sidebar / main inputs
st.header("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        "Location",
        options=sorted(location_categories),
        help="Select the area/locality in Bengaluru."       #help option
    )

    total_sqft = st.number_input(
        "Total Area (sqft)",
        min_value=300.0,
        max_value=10000.0,
        value=1200.0,
        step=50.0
    )

with col2:
    bhk = st.number_input(
        "BHK (Bedrooms)",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    bath = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

balcony = st.number_input(
    "Balconies",
    min_value=0,
    max_value=5,
    value=1,
    step=1
)

st.markdown("---")

if st.button("🔮 Predict Price"):
    with st.spinner("Calculating..."):
        predicted_price = predict_price(
            location=location,
            total_sqft=total_sqft,
            bath=bath,
            balcony=balcony,
            bhk=bhk
        )
    st.success(
        f"Estimated Price: **₹ {predicted_price:.2f} Lakhs**"
    )
    st.caption("Note: This estimate is based on historical Bengaluru housing data and a Linear Regression model.")

st.markdown("---")
st.write("Built with ❤️ using **Streamlit**, **Scikit-Learn**, and **Python**.")
st.write("Darshan")
