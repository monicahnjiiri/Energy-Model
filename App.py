import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load the trained model using a relative path
model_path = os.path.join(os.path.dirname(__file__), 'gradient_boosting_model.pkl')
model = joblib.load(model_path)

# Set up the title and description of the app
st.title("Energy Consumption Prediction App")
st.write("""
    This app predicts energy consumption based on building features like square footage, temperature, etc.
""")

# Define input fields for user input
building_type = st.selectbox('Building Type', ['Residential', 'Commercial', 'Industrial'])
square_footage = st.number_input('Square Footage', min_value=0, step=1)
num_occupants = st.number_input('Number of Occupants', min_value=0, step=1)
appliances_used = st.number_input('Appliances Used', min_value=0, step=1)
avg_temp = st.number_input('Average Temperature', min_value=0.0, step=0.1)
day_of_week = st.selectbox('Day of Week', ['Weekday', 'Weekend'])

# Prepare the input data for prediction
data = {
    'Building Type': [building_type],
    'Square Footage': [square_footage],
    'Number of Occupants': [num_occupants],
    'Appliances Used': [appliances_used],
    'Average Temperature': [avg_temp],
    'Day of Week': [day_of_week]
}

input_df = pd.DataFrame(data)

# Encode categorical features
input_df['Building Type'] = input_df['Building Type'].map({'Residential': 0, 'Commercial': 1, 'Industrial': 2})
input_df['Day of Week'] = input_df['Day of Week'].map({'Weekday': 0, 'Weekend': 1})

# Make predictions using the loaded model
prediction = model.predict(input_df)

# Display the prediction result
st.write(f"Predicted Energy Consumption: {prediction[0]:.2f} kWh")
