import pandas as pd
import numpy as np

def create_vibration_dataset():
    np.random.seed(42)
    total_windows = 52
    data_points_per_window = 100
    
    timestamps = pd.date_range(start="2026-07-12 11:00:00", periods=total_windows * data_points_per_window, freq="100ms")
    vibrations = np.random.normal(loc=0.0, scale=0.15, size=len(timestamps))
    
    anomaly_start = 25 * data_points_per_window
    anomaly_end = 38 * data_points_per_window
    vibrations[anomaly_start:anomaly_end] += np.random.uniform(0.6, 1.4, size=(anomaly_end - anomaly_start))
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "vibration_amplitude": vibrations,
        "window_index": np.repeat(np.arange(total_windows), data_points_per_window)
    })
    
    df.to_csv("vibration_data.csv", index=False)
    print("Successfully generated simulated 'vibration_data.csv'.")

if __name__ == "__main__":
    create_vibration_dataset()