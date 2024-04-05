import airsim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a new directory for this run
results_dir = f'results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)


sensor_data = []
# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


start_time = time.time()

while time.time() - start_time < 100:
    now = time.time()
    
    # Fetch sensor and state data
    imu_data = client.getImuData()
    barometer_data = client.getBarometerData()
    gps_data = client.getGpsData()
    state = client.simGetVehiclePose()
    position = state.position

    # Record sensor and state data
    data_entry = {
        'Time(s)': now - start_time,
        'X(m)': position.x_val,
        'Y(m)': position.y_val,
        'Z(m)': position.z_val,
        'Angular Velocity X(rad/s)': imu_data.angular_velocity.x_val,
        'Angular Velocity Y(rad/s)': imu_data.angular_velocity.y_val,
        'Angular Velocity Z(rad/s)': imu_data.angular_velocity.z_val,
        'Linear Acceleration X(ms^-2)': imu_data.linear_acceleration.x_val,
        'Linear Acceleration Y(ms^-2)': imu_data.linear_acceleration.y_val,
        'Linear Acceleration Z(ms^-2)': imu_data.linear_acceleration.z_val,
        'Orientation W': imu_data.orientation.w_val,
        'Orientation X': imu_data.orientation.x_val,
        'Orientation Y': imu_data.orientation.y_val,
        'Orientation Z': imu_data.orientation.z_val,
        'Barometer Altitude(m)': barometer_data.altitude,
        'Barometer Pressure(Pa)': barometer_data.pressure,
        'GPS Latitude(deg)': gps_data.gnss.geo_point.latitude,
        'GPS Longitude(deg)': gps_data.gnss.geo_point.longitude,
        'GPS Altitude(m)': gps_data.gnss.geo_point.altitude
    }

    sensor_data.append(data_entry)


client.armDisarm(False)
client.enableApiControl(False)

sensor_data_df = pd.DataFrame(sensor_data)
sensor_data_df.to_excel(os.path.join(results_dir,'sensor_grounded.xlsx'), index=False)
