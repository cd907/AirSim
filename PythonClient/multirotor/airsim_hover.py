import airsim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

# Function to check proximity between current position and target waypoint
def is_close(current, target, threshold=0.1):
    return np.linalg.norm(np.array([current.x_val - target.x_val, current.y_val - target.y_val, current.z_val - target.z_val])) < threshold

class PIDController:
    def __init__(self, kp_x=1, ki_x=0, kd_x=10, max_output_x=1, kp_y=1, ki_y=0, kd_y=10, max_output_y=1, kp_z=1, ki_z=0, kd_z=10, max_output_z=1):
        self.kp_x = kp_x
        self.ki_x = ki_x
        self.kd_x = kd_x
        self.integral_x = 0
        self.prev_error_x = 0
        self.max_output_x = max_output_x

        self.kp_y = kp_y
        self.ki_y = ki_y
        self.kd_y = kd_y
        self.integral_y = 0
        self.prev_error_y = 0
        self.max_output_y = max_output_y

        self.kp_z = kp_z
        self.ki_z = ki_z
        self.kd_z = kd_z
        self.integral_z = 0
        self.prev_error_z = 0
        self.max_output_z = max_output_z

    def update_controls(self, error_x, error_y, error_z, dt):
        # Update integral for X
        self.integral_x += self.ki_x * 0.5 * (error_x + self.prev_error_x) * dt
        self.integral_x = max(min(self.integral_x, self.max_output_x), -self.max_output_x)

        # Calculate derivative for X
        derivative_x = (error_x - self.prev_error_x) / dt if dt > 0 else 0

        # Calculate output for X
        control_x = self.kp_x * error_x + self.integral_x + self.kd_x * derivative_x
        control_x = max(min(control_x, self.max_output_x), -self.max_output_x)

        # Update integral for Y
        self.integral_y += self.ki_y * 0.5 * (error_y + self.prev_error_y) * dt
        self.integral_y = max(min(self.integral_y, self.max_output_y), -self.max_output_y)

        # Calculate derivative for Y
        derivative_y = (error_y - self.prev_error_y) / dt if dt > 0 else 0

        # Calculate output for Y
        control_y = self.kp_y * error_y + self.integral_y + self.kd_y * derivative_y
        control_y = max(min(control_y, self.max_output_y), -self.max_output_y)

        # Update integral for Z
        self.integral_z += self.ki_z * 0.5 * (error_z + self.prev_error_z) * dt
        self.integral_z = max(min(self.integral_z, self.max_output_z), -self.max_output_z)

        # Calculate derivative for Y
        derivative_z = (error_z - self.prev_error_z) / dt if dt > 0 else 0

        # Calculate output for Y
        control_z = self.kp_z * error_z + self.integral_z + self.kd_z * derivative_z
        control_z = max(min(control_z, self.max_output_z), -self.max_output_z)

        # Update previous errors
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_z = error_z

        return control_x, control_y, control_z
    
# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a new directory for this run
results_dir = f'results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)


# Initialize PID controllers for X and Y axes with separate parameters
pid_controller = PIDController(kp_x=0.5, ki_x=0, kd_x=0, max_output_x=10,
                               kp_y=0.5, ki_y=0, kd_y=0, max_output_y=10, kp_z=0.5, ki_z=0, kd_z=0, max_output_z=10)
# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


client.takeoffAsync()

# Target waypoint
target_waypoint = airsim.Vector3r(0, 0, -10)  # Assuming NED coordinates
sensor_data = []
start_time = time.time()

# Loop until the drone is close enough to the target waypoint
while not is_close(client.simGetVehiclePose().position, target_waypoint):
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

    if not is_close(position, target_waypoint):
        # use PID Control logic moving to the next waypoint asynchronously
        # Calculate position error in X-Y plane
        error_x = target_waypoint.x_val - position.x_val
        error_y = target_waypoint.y_val - position.y_val
        error_z = target_waypoint.z_val - position.z_val
        

        # Update PID controllers
        control_x, control_y, control_z = pid_controller.update_controls(error_x, error_y, error_z, dt=0.1)
        client.moveByVelocityAsync(vx=control_x, vy=control_y, vz=control_z, duration=0.1)

    time.sleep(0.1)

# Hover for 100 seconds at the target height
hover_start_time = time.time()
while time.time() - hover_start_time < 100:
    # Fetch sensor and state data
    imu_data = client.getImuData()
    barometer_data = client.getBarometerData()
    gps_data = client.getGpsData()
    state = client.simGetVehiclePose()
    position = state.position

    # Record sensor and state data
    data_entry = {
        'Time(s)': time.time() - start_time,
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

    time.sleep(0.1)

# Landing
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

sensor_data_df = pd.DataFrame(sensor_data)
sensor_data_df.to_excel(os.path.join(results_dir,'sensor_hover.xlsx'), index=False)

