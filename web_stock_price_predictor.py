import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
# st.set_page_config(page_title="Detailed Equity Analysis and Prediction App", layout="wide")
st.title("Detailed Equity Analysis and Prediction App",)
stock = st.text_input("Enter the Stock ID", "GOOG")
st.write("Find the stock symbol [here](https://finance.yahoo.com/).")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data")
st.write(google_data)

spliting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[spliting_len:])

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Moving Averages Plots
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_FOR_250_DAYS'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_FOR_250_DAYS'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_FOR_200_DAYS'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_FOR_200_DAYS'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_FOR_100_DAYS'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_FOR_100_DAYS'], google_data, 0))
# Calculate rolling volatility
google_data['Volatility'] = google_data['Close'].rolling(window=20).std()

# Plot volatility
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data['Volatility'], label="Rolling Volatility (20-day)", color='purple')
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Stock Price Volatility")
plt.legend()
st.pyplot(fig)


Scaler = MinMaxScaler(feature_range=(0, 1))
Scaled_data = Scaler.fit_transform(google_data[['Close']].dropna())

# Predict Future Prices
future_days = st.number_input("Enter the number of future days to predict", min_value=1, max_value=100, value=10)
last_100_days = Scaled_data[-100:]

# Prepare data for future predictions
x_future = [last_100_days]  # Start with the last 100 scaled values
future_predictions = []     # To store future predicted prices

for _ in range(future_days):
    # Predict the next day's price
    prediction = model.predict(np.array(x_future[-1]).reshape(1, -1, 1))
    future_predictions.append(prediction[0][0])  # Append the predicted value
    # Prepare the input for the next prediction
    next_input = np.append(x_future[-1][1:], prediction, axis=0)
    x_future.append(next_input)

# Rescale predictions back to original values
future_predictions = Scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Future dates
future_dates = [end + timedelta(days=i) for i in range(1, future_days + 1)]

# Ensure lengths match
assert len(future_dates) == len(future_predictions)

# Create DataFrame for future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
future_df.set_index('Date', inplace=True)

# Display Future Predictions
st.subheader("Future Price Predictions")
st.write(future_df)

# Plot original and predicted prices
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data.Close, label="Historical Prices")
plt.plot(future_df['Predicted Price'], label="Predicted Future Prices", linestyle='solid')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price Prediction")
plt.legend()
st.pyplot(fig)
