# Real-Time Anomaly Detection with Isolation Forest & ADWIN

Welcome to the Real-Time Anomaly Detection project! This system is designed to monitor continuous data streams and spot unusual patterns in real time, whether it's financial transactions, system metrics, or any other floating-point data. The core technologies behind this project are **Isolation Forest** for detecting anomalies and **ADWIN** for adapting to changes in data trends. Plus, we've included a real-time visualization to help you see what's going on as it happens.

## Project Overview

The main goal here is to track and identify anomalies in a live data stream. We simulate this stream with numbers that have regular patterns, some noise, and the occasional anomaly. Here's a quick look at what the project offers:

- **Isolation Forest**: This machine learning algorithm helps us spot anomalies in our data stream.
- **ADWIN (Adaptive Windowing)**: This tool adapts to changes in the data over time, adjusting our detection window dynamically.
- **Real-Time Visualization**: An interactive plot shows both the incoming data and any anomalies detected, along with how ADWIN is adjusting to changes.

## Features

1. **Anomaly Detection**:
   - Utilizes **Isolation Forest** to pinpoint outliers based on the score of each data point.
   - Operates using a sliding window, ensuring we handle streaming data effectively in real time.

2. **Concept Drift Handling**:
   - **ADWIN** dynamically adjusts its window size based on significant changes in data variance.
   - This means the system can keep up with evolving data trends.

3. **Real-Time Visualization**:
   - Visualizes the data stream live, marking anomalies in red.
   - Shows how ADWIN's mean adjusts over time with an orange dashed line.

## Dependencies

To get this project running, you'll need the following Python libraries:

- `numpy` for numerical computations
- `matplotlib` for creating visualizations
- `scikit-learn` for the Isolation Forest algorithm

Install these dependencies by running:

```bash
pip install -r requirements.txt
```

## How It Works

1. **Data Stream Simulation**:
   - A function called `data_stream` creates a flow of data points with regular patterns, noise, and random anomalies.

2. **Anomaly Detection**:
   - **Isolation Forest** is used to find outliers in the data stream. It processes data in a rolling window, meaning it updates with each new batch of data points.
   - Any data point with a negative anomaly score is flagged as an anomaly.

3. **Concept Drift Adaptation**:
   - **ADWIN** watches for changes in data variance and adjusts its window size accordingly.
   - This allows the system to adapt to shifts in data trends over time.

4. **Real-Time Visualization**:
   - The `matplotlib` animation updates in real time, showing the data stream and highlighting anomalies.
   - It also displays the mean from ADWIN, adjusted dynamically with an orange dashed line.

## Customization

Feel free to tweak the following parameters to fit your needs:

- **WINDOW_SIZE**: Adjusts the size of the sliding window for Isolation Forest.
- **CONTAMINATION**: Sets the expected proportion of anomalies in the data stream for Isolation Forest.
- **ADWIN_THRESHOLD**: Changes the sensitivity of ADWIN in detecting concept drift.

You can change these settings directly in the `main.py` file.

## Explanation of Key Algorithms

- **Isolation Forest**:
  - This algorithm identifies anomalies by isolating them using trees. Points that are separated early are considered anomalies.

- **ADWIN (Adaptive Windowing)**:
  - This method detects concept drift by adjusting the window size dynamically in response to significant changes in the data.

## Future Enhancements

Here are a few ideas for how the project could be improved:

- Adding more anomaly detection techniques for comparison, like One-Class SVM or DBSCAN.
- Implementing real-time alerts for detected anomalies via email or SMS.
- Extending the system to handle multivariate data streams for more complex scenarios.
