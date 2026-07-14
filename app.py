import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('hotel_demand_model.pkl')

model = load_model()

# Streamlit app
st.title("HotelVision")
st.write("AI-Powered Demand Forecasting Platform")

# Input fields
st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    date_input = st.date_input("Select Date", datetime.now())
    day_of_week = date_input.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    day_of_year = date_input.timetuple().tm_yday
    month = date_input.month

with col2:
    holiday_season = st.selectbox("Holiday Season", [0, 1], format_func=lambda x: "Yes" if x else "No")
    special_event = st.selectbox("Special Event", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Create cyclical features efficiently
sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
cos_day = np.cos(2 * np.pi * day_of_year / 365.25)

# Prediction
if st.button("Predict Demand"):
    # Prepare input data with minimal features
    input_data = pd.DataFrame({
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'holiday_season': [holiday_season],
        'special_event': [special_event],
        'day_of_year': [day_of_year]
    })
    


    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Room Demand: {prediction:.0f} rooms")
    
    # Display input summary
    st.subheader("Input Summary")
    st.write(f"Date: {date_input}")
    st.write(f"Holiday Season: {'Yes' if holiday_season else 'No'}")
    st.write(f"Special Event: {'Yes' if special_event else 'No'}")



# Batch prediction feature with optimizations
st.header("Batch Prediction")
st.write("Upload a CSV file with the required columns for batch predictions")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        # Read only necessary columns to save memory
        df = pd.read_csv(uploaded_file, usecols=[
            'day_of_week', 'is_weekend', 'holiday_season', 'special_event', 
            'month', 'day_of_year'
        ])
        
        # Add cyclical features
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        required_columns = [
            'day_of_week', 'is_weekend', 'holiday_season', 'special_event', 
            'month', 'sin_day', 'cos_day'
        ]
        
        if all(col in df.columns for col in required_columns):
            # Process in chunks if dataset is large
            if len(df) > 10000:
                st.info("Processing large dataset in chunks...")
                chunks = np.array_split(df, max(1, len(df) // 5000))
                predictions = []
                
                for i, chunk in enumerate(chunks):
                    st.write(f"Processing chunk {i+1}/{len(chunks)}")
                    predictions.extend(model.predict(chunk[required_columns]))
                
                df['predicted_demand'] = predictions
            else:
                # Process all at once for smaller datasets
                df['predicted_demand'] = model.predict(df[required_columns])
            
            st.success("Predictions completed!")
            
            # Show only a sample of results for large datasets
            if len(df) > 100:
                st.dataframe(df.head(100))
                st.write(f"... and {len(df) - 100} more rows")
            else:
                st.dataframe(df)
            
            # Download button for results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="hotel_demand_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"CSV file must contain columns: {required_columns}")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")


