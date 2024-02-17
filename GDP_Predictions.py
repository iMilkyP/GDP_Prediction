import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler 

st.title("GDP Predictor")

@st.cache_resource 
def load_my_model():
    model = load_model('SimpleRNN_Forecasting.h5') 
    return model

@st.cache_resource 
def load_scaler():
    scaler = MinMaxScaler()
    return scaler

model = load_my_model()
scaler = load_scaler()

def format_rnn_input(data_scaled, timesteps):
    x, y = [], []
    for i in range(timesteps, 105):
        x.append(data_scaled[i-timesteps:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file) 
    gdp_data = data['Total GDP Province']
    test = gdp_data.values.reshape(-1, 1)
    test2 = scaler.fit_transform(test)
    print(test2)
    # Data Scaling
    gdp_data_scaled = scaler.fit_transform(gdp_data.values.reshape(-1, 1)) 
    # Prepare data in RNN testing format
    timesteps = 2
    X_test, y_test = format_rnn_input(gdp_data_scaled, timesteps)
    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
    y_test = np.reshape(y_test, (-1, 1))
    
    # Generate predictions 
    predictions = scaler.inverse_transform(model.predict(X_test))
    predicitons_actual = scaler.inverse_transform(y_test)

    # Display Predictions Table
    st.header("Predictions: ")
    pred_df = pd.DataFrame({'Predicted': predictions.flatten(), 
                        'Actual': predicitons_actual.flatten()})
    st.table(pred_df) 

    # Display Test Graph (Include Calculation)
    st.header("Test Results: ")
    
    fig, ax = plt.subplots(figsize=(30, 10)) 
    
    ax.plot(predictions, label='y_pred', ls='--', lw = 2)
    ax.plot(predicitons_actual, label='y_test_actual')
    ax.legend()
    st.pyplot(fig)
    
    # Calculate and Display Metrics
    loss, accuracy = model.evaluate(X_test, y_test)
    rmse = sqrt(mean_squared_error(predicitons_actual, predictions))

    st.header("Evaluation Metrics")
    st.write(f"Loss: {loss:.4f}")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Test RMSE: {rmse:.2f}")
    
else:
    st.write("Upload a CSV file to get predictions.")