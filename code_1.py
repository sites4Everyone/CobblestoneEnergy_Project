import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from sklearn.ensemble import IsolationForest
import random

# Set parameters for the data stream and anomaly detection
WINDOW_SIZE = 50  # Rolling window for anomaly detection
CONTAMINATION = 0.05  # Expected proportion of anomalies
ADWIN_THRESHOLD = 0.1  # Threshold for detecting concept drift with ADWIN

# Queues to store the rolling data window and detected anomalies
data_window = deque(maxlen=WINDOW_SIZE)
anomalies_window = deque(maxlen=WINDOW_SIZE)

# A function to simulate a real-time data stream
def data_stream():
    time_step = 0
    while True:
        # Sinusoidal pattern to simulate seasonality
        seasonal_value = 10 * np.sin(2 * np.pi * time_step / 30)
        
        # Adding some random noise
        noise_value = random.uniform(-3, 3)
        
        # Simulate normal data points (seasonal + noise)
        data_point = seasonal_value + noise_value
        
        # Occasionally inject an anomaly (a rare large spike)
        if random.random() < 0.05:  # 5% chance of anomaly
            data_point += random.uniform(10, 20)
        
        yield data_point  # Yield one data point at a time to simulate streaming
        time_step += 1

# Function to detect anomalies using Isolation Forest
def detect_anomaly(data_point, isolation_forest_model):
    # Add the new data point to the rolling window
    data_window.append(data_point)
    
    # Once the window is full, apply the Isolation Forest model
    if len(data_window) == WINDOW_SIZE:
        # Train the Isolation Forest on the current window of data
        isolation_forest_model.fit(np.array(data_window).reshape(-1, 1))
        
        # Get the anomaly score for the current data point
        anomaly_score = isolation_forest_model.decision_function([[data_point]])[0]
        
        # If the score is negative, flag it as an anomaly
        if anomaly_score < 0:
            anomalies_window.append(data_point)
        else:
            anomalies_window.append(np.nan)  # No anomaly, append NaN for plotting
    else:
        anomalies_window.append(np.nan)  # Until the window is full, append NaN

# ADWIN (Adaptive Windowing) implementation for concept drift detection
class ADWIN:
    def __init__(self, threshold=ADWIN_THRESHOLD):
        self.window = deque()
        self.total = 0  # Track the total of the window values
        self.threshold = threshold  # Set the threshold for concept drift detection

    def update(self, value):
        # Add the new value to the window
        self.window.append(value)
        self.total += value
        
        # Only start detecting drift once we have more than 1 data point
        if len(self.window) > 1:
            mean = self.total / len(self.window)
            variance = sum((x - mean) ** 2 for x in self.window) / len(self.window)
            
            # Detect drift by comparing variance to the threshold
            if variance > self.threshold:
                # If drift is detected, shrink the window by removing older values
                self.window.popleft()
                self.total -= value

    def get_mean(self):
        # Return the current mean of the window
        return self.total / len(self.window) if len(self.window) > 0 else 0

# Setting up the real-time visualization
fig, ax = plt.subplots()
x_data, y_data, anomaly_data, adwin_mean_data = [], [], [], []
adwin = ADWIN()  # Instantiate the ADWIN class to monitor concept drift

# The update function for real-time plot animation
def update(frame_count):
    # Get the next data point from the simulated stream
    data_point = next(data_stream_gen)
    
    # Use the anomaly detection model to check if it's an anomaly
    detect_anomaly(data_point, isolation_forest_model)
    
    # Update ADWIN to monitor for concept drift
    adwin.update(data_point)
    adwin_mean = adwin.get_mean()  # Get the current mean of the ADWIN window
    
    # Append the new data for visualization
    x_data.append(frame_count)
    y_data.append(data_point)
    anomaly_data.append(anomalies_window[-1])  # Either an anomaly or NaN
    adwin_mean_data.append(adwin_mean)
    
    # Clear and redraw the plot
    ax.clear()
    ax.plot(x_data, y_data, label='Data Stream')
    ax.plot(x_data, adwin_mean_data, label='ADWIN Mean', linestyle='--', color='orange')
    ax.scatter(x_data, anomaly_data, color='red', label='Anomalies')
    
    # Set plot titles and labels
    ax.set_title("Real-Time Anomaly Detection with Isolation Forest & ADWIN")
    ax.set_xlabel("Time")
    ax.set_ylabel("Data Value")
    ax.legend()

# Set up the data stream generator and Isolation Forest model
data_stream_gen = data_stream()  # This will simulate the real-time data stream
isolation_forest_model = IsolationForest(contamination=CONTAMINATION)  # Anomaly detector

# Create the animation object, refreshing every 200 ms
ani = FuncAnimation(fig, update, interval=200)
plt.show()
