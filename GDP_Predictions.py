import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler 

# **Title:** Sets the title of the Streamlit application
st.title("GDP Prediction Tester")

# **Caching Functions:** Decorators for efficient model and scaler loading
@st.cache_resource 
def load_my_model():
    """Loads the pre-trained SimpleRNN model from the 'SimpleRNN_Forecasting.h5' file.

    Returns:
        The loaded Keras model.
    """
    model = load_model('SimpleRNN_Forecasting.h5') 
    return model

@st.cache_resource 
def load_scaler():
    """Loads a MinMaxScaler object for data normalization.

    Returns:
        The fitted MinMaxScaler object.
    """
    scaler = MinMaxScaler()
    return scaler

# **Global Variables:** Loads model and scaler outside of main loop for efficiency
model = load_my_model()
scaler = load_scaler()

def format_rnn_input(data_scaled, timesteps):
    """Prepares data for the RNN model, creating overlapping sequences.

    Args:
        data_scaled: The scaled GDP data (NumPy array).
        timesteps: The number of previous data points to include in each sequence.

    Returns:
        A tuple containing:
            X_test: NumPy array of input sequences.
            y_test: NumPy array of target values.
    """
    x, y = [], []
    for i in range(timesteps, 105):
        x.append(data_scaled[i-timesteps:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)

# **File Uploader:** Allows user to select a CSV file 
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

min_year = 2018
max_year = 2023

selected_years = st.slider(
    "Select a year range:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

year1, year2 = selected_years

if not (min_year <= year1 <= year2 <= max_year):
    st.error("Please enter valid years within the specified range.")
    year_range_str = f"{year1} - {year2}"
    # **Conditional Block:** Executes prediction logic if a file is uploaded
    if uploaded_file is not None:
        # **Data Loading:**
        data = pd.read_csv(uploaded_file)
        year = data[data['Years'] == year_range_str] # Filter for the relevant year
        gdp_data = year['Total GDP Province']
        timesteps = 2
        
        # **Data Scaling:**
        gdp_data_scaled = scaler.fit_transform(gdp_data.values.reshape(-1, 1)) 
        X_test, y_test = format_rnn_input(gdp_data_scaled, timesteps)
        
        # **Reshaping for RNN:**
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
        y_test = np.reshape(y_test, (-1, 1))
        
        # **Generating Predictions:**
        predictions = scaler.inverse_transform(model.predict(X_test))
        predicitons_actual = scaler.inverse_transform(y_test)

        # **Predictions Table:**
        st.header("Predictions Table: ")
        pred_df = pd.DataFrame({'Predicted': predictions.flatten(), 
                            'Actual': predicitons_actual.flatten()})
        st.table(pred_df) 

        # **Test Results Header:**
        st.header("Test Results: ")
        
        # **Predictions Graph:**
        st.header("Predictions Graph: ")  
        pred_df = pd.DataFrame({'Predicted': predictions.flatten(), 
                                'Actual': predicitons_actual.flatten()})
        st.line_chart(pred_df) 
        
        # **Evaluation Metrics:**
        loss, accuracy = model.evaluate(X_test, y_test)
        rmse = sqrt(mean_squared_error(predicitons_actual, predictions))

        st.header("Evaluation Metrics")
        st.write(f"Loss: {loss:.4f} or {loss * 100:.2f}%")
        st.write(f"Accuracy: {accuracy:.4f} or {accuracy * 100:.2f}%")
        st.write(f"Test RMSE: {rmse:.2f}")
        
    else:
        st.write("Upload a CSV file to get predictions.")