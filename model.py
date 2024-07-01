import os
import pandas as pd
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

ist_timezone = pytz.timezone('Asia/Kolkata')

# Read and preprocess data
n = 4
N = int(n / 2)
directory = 'datasets'
files = [file for file in os.listdir(directory) if file.endswith('.csv')]
files.sort()
train_files = files[:n]
dfs = []

for file in train_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)

# Data Normalization
scaler = MinMaxScaler()
data[['open', 'high', 'low', 'close']] = scaler.fit_transform(data[['open', 'high', 'low', 'close']])

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Data Preparation for LSTM
def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step)])
        y.append(dataset[i + time_step])
    return np.array(X), np.array(y)

time_step = 90
X_train, y_train = create_dataset(train_data[['open', 'high', 'low', 'close']].values, time_step)
X_test, y_test = create_dataset(test_data[['open', 'high', 'low', 'close']].values, time_step)

# Print shapes for debugging
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 4)

# Model Creation and Training
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=4))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# ModelCheckpoint
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

model.fit(X_train, y_train, epochs=940*N, batch_size=32, validation_split=0.2, callbacks=[model_checkpoint])

# Model Evaluation
predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)

# Inverse transform predictions
predicted_train = scaler.inverse_transform(predicted_train)
predicted_test = scaler.inverse_transform(predicted_test)
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

# Calculate R-squared for train and test sets
r2_train = r2_score(y_train, predicted_train, multioutput='uniform_average')
r2_test = r2_score(y_test, predicted_test, multioutput='uniform_average')

print("R-squared (Train): {:.2f}%".format(r2_train * 100))
print("R-squared (Test): {:.2f}%".format(r2_test * 100))

# Calculate additional metrics for test set
mae_test = mean_absolute_error(y_test, predicted_test)
rmse_test = mean_squared_error(y_test, predicted_test, squared=False)

print("Mean Absolute Error (Test): {:.4f}".format(mae_test))
print("Root Mean Squared Error (Test): {:.4f}".format(rmse_test))

# Visualization of Predictions (only predicted values for test data)
plt.figure(figsize=(14, 5))
plt.plot(test_data.index[time_step:], predicted_test[:, 3], color='red', label='Predicted Close')
plt.plot(test_data.index[time_step:], y_test[:, 3], color='blue', label='Actual Close')
plt.title('Stock Price Prediction - LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Predict Next Full Trading Day Data Minute by Minute
last_data_points = test_data[['open', 'high', 'low', 'close']].values[-time_step:]
X_next_day = last_data_points.reshape(1, time_step, 4)

num_minutes = int(6 * 60 + 16)  # Convert to integer

next_day_predictions = []
next_day_timestamps = []

next_trading_day = test_data.index[-1] + pd.Timedelta(days=1)
start_timestamp = pd.Timestamp(next_trading_day.year, next_trading_day.month, next_trading_day.day, 3, 45)

for i in range(num_minutes):
    pred = model.predict(X_next_day)
    next_day_predictions.append(pred[0])
    next_timestamp = start_timestamp + pd.Timedelta(minutes=i)
    next_day_timestamps.append(next_timestamp)
    X_next_day = np.append(X_next_day[:, 1:, :], np.expand_dims(pred, axis=1), axis=1)

next_day_predictions = scaler.inverse_transform(np.array(next_day_predictions))

print("Next Full Trading Day Predicted Values (Minute by Minute):")
print(next_day_predictions)

# Save predictions to CSV
next_day_df = pd.DataFrame({
    'timestamp': [int(ts.timestamp()) for ts in next_day_timestamps],
    'open': next_day_predictions[:, 0],
    'high': next_day_predictions[:, 1],
    'low': next_day_predictions[:, 2],
    'close': next_day_predictions[:, 3]
})
# output_directory = ""
# os.makedirs(output_directory, exist_ok=True)
# output_file = os.path.join(output_directory, "output.csv")
next_day_df.to_csv("output.csv", index=False)

final_predicted_values = next_day_predictions[-1]
print("Final Predicted Closing Price of the Stock:", final_predicted_values[3])
