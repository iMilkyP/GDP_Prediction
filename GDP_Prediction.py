import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

# Function to load your model (customize path as needed)
@st.cache_resource 
def load_my_model():
    model = load_model('SimpleRNN_Forecasting.h5') 
    print(model)
    return model

model = load_my_model()

def preprocess_data(data):
    # Drop non-numeric columns and scale numeric columns
    numeric_data = data.drop(columns=['Years', 'Sectors', 'Province'])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    return scaled_data

# Prediction Logic
def make_predictions(model, data, scaler):
    scaled_data = preprocess_data(data)[0]  # Use only the scaled data
    predictions = model.predict(scaled_data)
    # Inverse scaling is optional depending on presentation preference
    unscaled_predictions = scaler.inverse_transform(predictions)  
    return predictions, unscaled_predictions

def main():
    st.title("GDP Predictor")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)
        
        st.write("Predictions:")
        
        year1819 = data[data['Years'] == '2018-2019']
        year1920 = data[data['Years'] == '2019-2020']
        year2021 = data[data['Years'] == '2020-2021']
        year2122 = data[data['Years'] == '2021-2022']
        train_data = pd.concat([year1819['Total GDP Province'], year1920['Total GDP Province'], year2021['Total GDP Province'], year2122['Total GDP Province']])
        
        train = train_data.values.reshape(-1,1)
        scaler = MinMaxScaler()
        scaled_trainset = scaler.fit_transform(train)
        
        print(scaled_trainset)
        
        # Plot the data
        st.subheader("Train Data")
        st.line_chart(pd.DataFrame({'Train Data': train.flatten()}))
        
        x_train = []
        y_train = []
        step = 2

        for i in range(step, 315):
            x_train.append(scaled_trainset[i-step:i,0])
            y_train.append(scaled_trainset[i,0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
        y_train.reshape(y_train.shape[0],1)
        
        model = Sequential()

        model.add(
            SimpleRNN(units = 50, return_sequences= True, input_shape = (x_train.shape[1],1), activation='relu'))

        model.add(
            Dropout(0.2))

        model.add(
            SimpleRNN(units = 50, activation='relu'))

        model.add(
            Dropout(0.2)
                    )
        model.add(
            Dense(units = 1, activation='relu'))
        
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        history = model.fit(x_train, y_train, epochs = 50, batch_size=8)
        
        # Plot train loss
        st.subheader("Train Loss")
        st.line_chart(history.history['loss'])
        
        # Plot train accuracy
        st.subheader("Train Accuracy")
        st.line_chart(history.history['accuracy'])
        
        y_pred = model.predict(x_train)
        y_pred = scaler.inverse_transform(y_pred.reshape(1,-1))
        
        y_train = scaler.inverse_transform(y_train.reshape(1,-1))
        y_train = np.reshape(y_train, (313,1))
        y_pred = np.reshape(y_pred,(313,1))
        
        # Plot y_pred and y_train
        st.subheader("y_pred vs y_train")
        st.line_chart(pd.DataFrame({'y_pred': y_pred.flatten(), 'y_train': y_train.flatten()}))
        
        loss, accuracy = model.evaluate(x_train, y_train)
        st.write(f"Loss: {loss}, Accuracy: {accuracy}")
        rmse = sqrt(mean_squared_error(y_train, y_pred))
        st.write(f"Test RMSE: {rmse}")

    else:
        st.write("Upload a CSV file to get predictions.")

if __name__ == '__main__':
    main()