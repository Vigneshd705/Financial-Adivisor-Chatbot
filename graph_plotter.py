import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def plot_comparison(actual_file, predicted_file):
    # Read actual and predicted data from CSV files
    actual_data = pd.read_csv(actual_file)
    predicted_data = pd.read_csv(predicted_file)

    # Convert timestamp to datetime
    actual_data['timestamp'] = pd.to_datetime(actual_data['timestamp'], unit='s')
    predicted_data['timestamp'] = pd.to_datetime(predicted_data['timestamp'], unit='s')

    # Set timestamp as index
    actual_data.set_index('timestamp', inplace=True)
    predicted_data.set_index('timestamp', inplace=True)

    # Plotting comparison
    plt.figure(figsize=(14, 7))

    # Plot actual values
    plt.plot(actual_data.index, actual_data['high'], label='Actual High', color='blue', linestyle='-')
    plt.plot(actual_data.index, actual_data['low'], label='Actual Low', color='green', linestyle='-')
    plt.plot(actual_data.index, actual_data['open'], label='Actual Open', color='purple', linestyle='-')
    plt.plot(actual_data.index, actual_data['close'], label='Actual Close', color='orange', linestyle='-')

    # Plot predicted values
    plt.plot(predicted_data.index, predicted_data['high'], label='Predicted High', color='blue', linestyle='--')
    plt.plot(predicted_data.index, predicted_data['low'], label='Predicted Low', color='green', linestyle='--')
    plt.plot(predicted_data.index, predicted_data['open'], label='Predicted Open', color='purple', linestyle='--')
    plt.plot(predicted_data.index, predicted_data['close'], label='Predicted Close', color='red', linestyle='--')

    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate accuracy metrics
    metrics = {}
    for column in ['high', 'low', 'open', 'close']:
        # Ensure columns exist in both actual and predicted dataframes
        if column in actual_data.columns and column in predicted_data.columns:
            r2 = r2_score(actual_data[column], predicted_data[column]) * 100  # Convert R-squared to percentage
            mae = mean_absolute_error(actual_data[column], predicted_data[column])
            rmse = mean_squared_error(actual_data[column], predicted_data[column], squared=False)
            metrics[column] = {'R-squared': r2, 'MAE': mae, 'RMSE': rmse}
        else:
            print(f"Column '{column}' not found in both actual and predicted dataframes.")

    # Display accuracy metrics
    for column, metric in metrics.items():
        print(f"Metrics for {column.capitalize()}:")
        print(f"R-squared: {metric['R-squared']:.2f}%")  # Print R-squared as percentage
        print(f"MAE: {metric['MAE']:.4f}")
        print(f"RMSE: {metric['RMSE']:.4f}")
        print()


actual_file = 'datasets/2023-06-23.csv'  # Replace with your actual data file path
predicted_file = 'output.csv'  # Replace with your predicted data file path

plot_comparison(actual_file, predicted_file)