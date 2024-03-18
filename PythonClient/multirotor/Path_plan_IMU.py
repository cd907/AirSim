import airsim
import numpy as np
import time
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to check proximity between current position and target waypoint
def is_close(current, target, threshold=1.0):
    return np.linalg.norm(np.array([current.x_val - target.x_val, current.y_val - target.y_val, current.z_val - target.z_val])) < threshold

# Function to add Gaussian noise to accelaration data
def add_noise(data, mean=0.0, std_dev=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    data_array = np.array([data.x_val, data.y_val, data.z_val])

    # Add random Gaussian noise
    noisy_data_array = data_array + np.random.normal(mean, std_dev, data_array.shape)
    noisy_data_vector3r = airsim.Vector3r(noisy_data_array[0], noisy_data_array[1], noisy_data_array[2])

    return noisy_data_vector3r

def calculate_desired_pitch(z1, z2, horizontal_distance):
    if horizontal_distance > 0:
        return math.atan2(z2 - z1, horizontal_distance)
    return 0

class PIDController:
    def __init__(self, kp_val=0.1, ki_val=0.01, kd_val=0.01,
                 min_output_val=-1, max_output_val=1):
        self.kp = kp_val
        self.ki = ki_val
        self.kd = kd_val
        self.integral = 0
        self.prev_error = 0
        self.min_output = min_output_val
        self.max_output = max_output_val

    
    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Define waypoints in mission (x, y, z in meters)
waypoints = [
    airsim.Vector3r(0, 0, -10),
    airsim.Vector3r(10, 0, -10),
    airsim.Vector3r(10, 10, -10),
    airsim.Vector3r(0, 10, -10),
    airsim.Vector3r(0, 0, -10)
]

csv_file_name = 'simulation_results.csv'
lidar_data_dir = "lidar_data"
os.makedirs(lidar_data_dir, exist_ok=True)

# Initialize PID controllers for X, Y, Z axes
pid_yaw = PIDController()
pid_pitch = PIDController()

# Define noise variances for different simulations
noise_variances = [0.0, 0.01, 0.1, 1.0]
results = []

for i, variance in enumerate(noise_variances):
    flight_path = []
    total_distance = 0
    waypoint_distances = []
    collision_count = 0
    start_time = time.time()

    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Takeoff
    client.takeoffAsync().join()
    
    for idx, waypoint in enumerate(waypoints):

        last_time = time.time()
        prev_position = client.simGetVehiclePose().position.to_numpy_array()
        # Continuously sample state until the drone is close to the waypoint
        while not is_close(client.simGetVehiclePose().position, waypoint):

            now = time.time()
            dt = now - last_time
            last_time = now

            # Get current state
            state = client.simGetVehiclePose()
            position = state.position
            orientation = state.orientation  # Quaternion
            roll, pitch, yaw = airsim.to_eularian_angles(orientation)

             # Add Gaussian noise to drone's current position data
            noisy_position = add_noise(position, mean=0.0, std_dev=np.sqrt(variance), seed=42)  # Adjust mean and std_dev as needed

            # Calculate accelarator error in X-Y plane
            error_x = waypoint.x_val - noisy_position.x_val
            error_y = waypoint.y_val - noisy_position.y_val

            # Calculate horizontal distance from current point to target
            horizontal_distance = math.sqrt((error_x) ** 2 + (error_y) ** 2)

            # Add noise to roll, pitch, yaw
            noisy_orientation = add_noise(orientation, mean=0.0, std_dev=np.sqrt(variance), seed=42)

            # Record position and orientation
            flight_path.append((position, orientation))

            # Calculate distance traveled since last position
            distance = np.linalg.norm(position.to_numpy_array() - prev_position)
            total_distance += distance
            prev_position = position.to_numpy_array()

            # use PID Control logic on angle level and moving to the next waypoint asynchronously

            # Calculate desired yaw and pitch
            desired_yaw = math.atan2(error_x, error_y)
            desired_pitch = calculate_desired_pitch(noisy_position.z_val, waypoint.z_val, horizontal_distance)

            # Update PID controllers
            # Calculate errors on yaw and pitch
            error_yaw = desired_yaw - yaw
            error_pitch = desired_pitch - pitch
            
            control_yaw = pid_yaw.update(error_yaw, dt)
            control_pitch = pid_pitch.update(error_pitch, dt)
        
        # regulate the velocity in X,Y axis, To BE completed!
            # max(min_value, min(val, max_value))

            client.moveByRollPitchYawZAsync(roll=0, pitch=control_pitch, yaw=control_yaw, z=waypoint.z_val, duration=1)

            # Check for collision
            collision_info = client.simGetCollisionInfo()

            if collision_info.has_collided:
                print("Collision detected!")
                collision_count += 1
            
            # Sleep for a short duration to avoid excessive sampling
            time.sleep(0.1)

        # Calculate distance from waypoint (considering it reached)
        waypoint_distance = np.linalg.norm(client.simGetVehiclePose().position.to_numpy_array() - waypoint.to_numpy_array())
        waypoint_distances.append(waypoint_distance)

        # # Retrieve LiDAR data at current position
        # lidar_data = client.getLidarData()
        # if len(lidar_data.point_cloud) < 3:
        #     continue
        # points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        
        # # Save LiDAR data to a file
        # lidar_filename = os.path.join(lidar_data_dir, f"waypoint_{idx+1}_lidar_data.csv")
        # np.savetxt(lidar_filename, points, delimiter=",", fmt='%f', header='x,y,z', comments='')

        # print(f"Reached waypoint {idx+1}, LiDAR data saved to {lidar_filename}")

    # Land
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    total_time = time.time() - start_time
    average_waypoint_distance = sum(waypoint_distances) / len(waypoint_distances) if waypoint_distances else 0

    # print(f"Total Distance Traveled: {total_distance} meters")
    # print(f"Total Flight Time: {total_time} seconds")
    # print(f"Collision Count: {collision_count}")
    # print(f"Average Distance from Waypoints: {average_waypoint_distance} meters")
    results.append({'Noise Variance': variance,
        'Total Distance Traveled (m)': total_distance,
        'Total Flight Time (s)': total_time,
        'Collision Count': collision_count,
        'Average Distance from Waypoints (m)': average_waypoint_distance})
    
    # Extracting X, Y, Z coordinates
    x_vals = [pos.x_val for pos, _ in flight_path]
    y_vals = [pos.y_val for pos, _ in flight_path]
    z_vals = [pos.z_val for pos, _ in flight_path]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the flight path as a scatter plot
    ax.scatter(x_vals, y_vals, z_vals, c='blue', s=10) 

    # Highlight the start point in green
    ax.scatter(x_vals[0], y_vals[0], z_vals[0], c='green', s=50, label='Start Point')

    # Highlight the end point in red
    ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], c='red', s=50, label='End Point')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Flight Path Visualization with Noise Variance {variance}')
    plt.savefig(f'flight_path_simulation_{i+1}.png')
    plt.clf()


df = pd.DataFrame(results)
df.to_excel('simulation_results.xlsx', index=False)

# Print recorded flight path
# print("Flight path:")
# for pos, orient in flight_path:
#     print(f"Position: ({pos.x_val}, {pos.y_val}, {pos.z_val}), Orientation (quaternion): ({orient.x_val}, {orient.y_val}, {orient.z_val}, {orient.w_val})")


# Plotting the flight path
# ax.plot(x_vals, y_vals, z_vals, marker='o')


