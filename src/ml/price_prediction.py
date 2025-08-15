import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

logger = logging.getLogger(__name__)

def create_lstm_model(input_shape):
    """Create an LSTM model for price prediction."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data: pd.DataFrame, lookback: int = 60):
    """Prepare data for LSTM model."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def train_lstm_model(data: pd.DataFrame, lookback: int = 60):
    """Train an LSTM model for price prediction."""
    try:
        X, y, scaler = prepare_data(data, lookback)
        model = create_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32)
        return model, scaler
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return None, None

def predict_price(model, data: pd.DataFrame, scaler, lookback: int = 60):
    """Predict the next price using the LSTM model."""
    try:
        last_data = data['Close'].values[-lookback:]
        scaled_data = scaler.transform(last_data.reshape(-1, 1))
        X = np.array([scaled_data])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predicted_price = model.predict(X)[0, 0]
        predicted_price = scaler.inverse_transform([[predicted_price]])[0, 0]
        return predicted_price
    except Exception as e:
        logger.error(f"Error predicting price: {e}")
        return None
