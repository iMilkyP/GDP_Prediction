import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler 
import keras
import tensorflow as tf


print("KERAS VERSION:", keras.__version__)
print("TENSORFLOW VERSION:", tf.__version__)

# Styling
st.set_page_config(layout="wide")
st.markdown("""
<style>
.stButton > button {
    background-color: #6930C3;
    text-color: white;
    width: 150px;
    height: 50px;            
    font-size: 18px;         
    border: none;           
    border-radius: 8px;      
    padding: 10px 20px;
    text-align: center; 
    display: block; 
    margin: auto;  
}
/* Hover Effect */
.stButton > button:hover {
    outline: 1px solid #64DFDF; 
    outline-offset: 2px;
}                     
</style>
""", unsafe_allow_html=True)

st.text("")

# Welcome
st.title("Welcome to the GDP Growth Rate Predictor for CAR")

st.text("")

st.header("Instruction")
st.text("1.Have your CSV file ready. ")
st.text('2.Click "Upload CSV File" and choose your file.')
st.text('3.Click "Predict GDP"')
st.text("4.Your predicted GDP will be shown ")

st.text("")

st.header("CSV Template")
st.text("Please follow the column labels from the format, label the column you would like to predict as 'Total GDP per Province'.")
#st.text("")
#st.text("Click the button to download the CSV template")
# Model and scaler loading 
@st.cache_resource 
def load_my_model():
    print("I ran in load model")
    model = load_model('SimpleRNN_Forecasting.h5') 
    return model



@st.cache_resource 
def load_scaler():
    scaler = MinMaxScaler()
    return scaler

# Loads model and scaler
model = load_my_model()
scaler = load_scaler()

print(model)

def format_rnn_input(data_scaled):
    x, y = [], []
    steps = 2
    for i in range(steps, len(data_scaled)):
        x.append(data_scaled[i-steps:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)

# CSV file template
template_csv_file = "GDP_Predictor_Template.csv"

# Read the template file's content
with open(template_csv_file, "rb") as file:
    csv_data = file.read()

# Create the download button
st.download_button(
    label="Download CSV Template",
    data=csv_data,
    file_name="GDP_Predictor_Template.csv",  
    mime="text/csv"
)

print(template_csv_file)

st.header("Upload CSV file")
# Allows user to select a CSV file 
uploaded_file = st.file_uploader("Upload your file here: ", type=['csv'])

# Executes prediction logic if a file is uploaded

if st.button("Predict GDP"):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Display the DataFrame as a table
        with st.expander("CSV File Preview:"):
            df_no_index = data.copy()
            df_no_index.set_index('Years', inplace=True)
            st.dataframe(df_no_index)
            year2019 = data[data['Years'] == '2018-2019']
            year2020 = data[data['Years'] == '2019-2020']
            year2021 = data[data['Years'] == '2020-2021']
            year2022 = data[data['Years'] == '2021-2022']
            gdp_data = year2022['Total GDP Province']

        
        col1, col2 = st.columns(2)

        # Data Scaling
        gdp_data_scaled = scaler.fit_transform(gdp_data.values.reshape(-1, 1)) 
        X_test, y_test = format_rnn_input(gdp_data_scaled)
            
        # Reshaping
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
        y_test = np.reshape(y_test, (-1, 1))
                
        # Generate Predictions
        predictions = scaler.inverse_transform(model.predict(X_test))  
        predicitons_actual = scaler.inverse_transform(y_test)
            
        # Test Result
        #st.header("---")

        # Prediction Graph
        #st.header("Accuracy Graph: ")  
        pred_df = pd.DataFrame({'Predicted': predictions.flatten(), 
                                        'Actual': predicitons_actual.flatten()})
        #st.line_chart(pred_df) 
            
        input_data_2023 = gdp_data[-8:].values.reshape(-1, 1) 
        scaled_input_data_2023 = scaler.transform(input_data_2023)
        X_2023 = np.reshape(scaled_input_data_2023, (1, len(input_data_2023), 1)) 

        predictions_2023 = [    ]
            

        for _ in range(12):  # 12 Months
            prediction_2023 = model.predict(X_2023)
            actual_prediction_2023 = scaler.inverse_transform(prediction_2023)
            predictions_2023.append(actual_prediction_2023)

        # Evaluation Metric
        loss, accuracy = model.evaluate(X_test, y_test)
        rmse = sqrt(mean_squared_error(predicitons_actual, predictions))
            
            #st.header("Evaluation Metrics")
            #st.write(f"Loss: {loss:.4f} or {loss * 100:.2f}%")
            #st.write(f"Accuracy: {accuracy:.4f} or {accuracy * 100:.2f}%")
            #st.write(f"Test RMSE: {rmse:.2f}")
            
            # Aggregate
        predicted_gdp_2023 = np.array(predictions_2023).mean() 
            
        year_data = {
            '2018-2019': year2019['Total GDP Province'],
            '2019-2020': year2020['Total GDP Province'],
            '2020-2021': year2021['Total GDP Province'],
            '2021-2022': year2022['Total GDP Province'],
            '2022-2023': predicted_gdp_2023
        }

        desired_years = ['2018-2019', '2019-2020', '2020-2021', '2021-2022']
        filtered_year_data = {year: year_data[year] for year in year_data if year in desired_years}    
        # Create a dictionary to store yearly GDP means
        year_means = {}
        for year, gdp_data in filtered_year_data.items():
            year_means[year] = gdp_data.mean()
            
        # Convert data into a Pandas DataFrame
        df = pd.DataFrame(year_means, index=['GDP'])
        
            # Display the table in Streamlit
            #st.table(df)
            
            
        
        # Convert means into a DataFrame for plotting
        df = pd.DataFrame({'Actual GDP growth (%)': year_means.values()}, index=year_means.keys())
        p_df = pd.DataFrame({'Year': ['2022-2023'], 'Predicted GDP growth(%)': [predicted_gdp_2023]})
                # Convert index to a column named "Year"
        df = df.reset_index() 
        df.rename(columns={'index': 'Year'}, inplace=True)
        with col2:
            # Line chart for yearly mean GDP
            st.header("Yearly GDP growth")
            df_no_index = df.copy()  # Create a copy
            df_no_index.set_index('Year', inplace=True)  # Set 'Year' as the index (removing the old index column)
            st.write(df_no_index)  # Display the modified DataFrame

            # Display the table
            st.header("Predicted GDP") 
            p_no_index = p_df.copy()
            p_no_index.set_index('Year', inplace=True)
            st.write(p_no_index)

            df_combined = pd.concat([df, p_df], ignore_index=True)
            #dfc_no_index = df_combined.copy()
            #dfc_no_index.set_index('Year', inplace=True)
            #st.write(dfc_no_index)

            csv = df_combined.to_csv(index=False)
            st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='gdp_data.csv',
            mime='text/csv'
        )

            
        

            st.write()

            st.write("""
            <span style="font-size: 24px;">Predicted GDP Growth Rate for 2022-2023: </span><strong><span style="font-size: 24px; color: #64DFDF">{:.2f}%</span></strong>
            """.format(predicted_gdp_2023), unsafe_allow_html=True)
            # Calculate GDP growth from previous year (as a percentage)
            previous_year_gdp = year2022['Total GDP Province'].mean()  # Adjust if needed 
            gdp_growth_percent = ((predicted_gdp_2023 - previous_year_gdp))

                # Generate conclusion statement
            if gdp_growth_percent > 0:
                conclusion = "Based on the prediction, there is an expected increase of approximately {:.2f}% in GDP for 2022-2023.".format(gdp_growth_percent)
            elif gdp_growth_percent == 0:
                conclusion = "Based on the prediction, there are no expected changes in GDP for 2022-2023.".format(gdp_growth_percent)
            else:
                conclusion = "Based on the prediction, there is an expected decrease of approximately {:.2f}% in GDP for 2022-2023.".format(abs(gdp_growth_percent))

            st.write("")    

            # Display the conclusion
            st.write(conclusion)

            st.write("")
        with col1:
            st.header("CAR GDP Growth")
            st.line_chart(df, x='Year', y='Actual GDP growth (%)')   
                
                # Table
                #st.header("Predictions Table: ")
                #pred_df = pd.DataFrame({'Predicted': predictions.flatten(), 
                                        #'Actual': predicitons_actual.flatten()})
                #st.table(pred_df) 
                    
    else:
        st.write("Upload a CSV file to get predictions.")
        