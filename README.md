# GDP Prediction Tester
- This Streamlit application enables you to predict GDP values using a pre-trained SimpleRNN model. Here's how to use it and a short overview of the project.

# Pre-requisites
- Python 3.11
## Required libraries:
- streamlit
- pandas
- numpy
- tensorflow (including Keras)
- scikit-learn (sklearn)
- matplotlib

Install the required libraries using pip:
```
pip install -r requirements.txt
```

# Running the Application
1. Download or clone this project.

2. Ensure the model file SimpleRNN_Forecasting.h5 is in the same directory.

3. Navigate to the project directory in your terminal.

4. Run the command:

```
streamlit run GDP_Predictions.py 
```

This launches the application in your web browser.

# How to Use
**1.  Upload a CSV file:** Use the "Upload CSV file" button and select a CSV file with the following structure:

- A column named "Years" containing years formatted as 'YYYY-YYYY' (e.g., '2021-2022').
- A column named "Total GDP Province" containing the corresponding GDP values.

**2. Predictions:** The application will process your file and display:

- A table showing predicted GDP values alongside actual values.
- A line chart visualizing the prediction results.
- Evaluation metrics: Loss, Accuracy, and Root Mean Squared Error (RMSE).

# Project Overview
- **Data Preprocessing:**

    - The application extracts GDP data for the year '2021-2022' from your CSV file.
    - It uses MinMaxScaler for normalization.

- **Model:**

    - A trained SimpleRNN model (SimpleRNN_Forecasting.h5) is used for predictions.
    - The model takes sequences of previous GDP values (lookback window configurable).

- **Visualization & Metrics:**

    - Streamlit components display a table, a line chart, and calculated metrics.

# Notes

- **Accuracy**:

    - The provided 'accuracy' metric might require adjustment if not appropriate for regression tasks.
    - Consider investigating alternative regression metrics (e.g., R-squared, Mean Absolute Percentage Error).


- **Model Improvement:** Experiment with hyperparameters, RNN model architecture, or additional data to potentially enhance predictions.


## Feel free to adapt and experiment!