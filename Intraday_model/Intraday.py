import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_intraday_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),  # LSTM units for capturing short-term patterns
        Dropout(0.2),  # Dropout to prevent overfitting
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),  # Dense layer for additional learning capacity
        Dense(1)  # Output layer for price prediction
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile model with Adam optimizer and MSE loss
    return model

def prepare_intraday_data(data):
    if len(data) < 60:
        raise ValueError("At least 60 data points are required for intraday trading.")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])  # Use the last 60 minutes for prediction
        y_train.append(scaled_data[i, 0])
    
    x_train = np.array(x_train).reshape(-1, 60, 1)  # Reshape for LSTM input
    return x_train, np.array(y_train), scaler

def predict_intraday_price(model, last_60_minutes, scaler):
    last_60_minutes_scaled = scaler.transform(last_60_minutes.reshape(-1, 1))
    x_test = np.array([last_60_minutes_scaled]).reshape(1, 60, 1)

    predicted_price = model.predict(x_test, verbose=0)  # Suppress output for faster execution
    return scaler.inverse_transform(predicted_price.reshape(-1, 1))[0][0]  # Return the predicted price in original scale
