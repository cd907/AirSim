import airsim
import numpy as np
import time
from datetime import datetime
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to check proximity between current position and target waypoint
def is_close(current, target, threshold=0.1):
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

def multiplicative_noise(value, mean=1.0, std_dev=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise_factor = np.random.normal(mean, std_dev)
    return value * noise_factor


class PIDController:
    def __init__(self, kp_val=0.25, ki_val=0.0, kd_val=0.0,
                max_output_val=10):
        self.kp = kp_val
        self.ki = ki_val
        self.kd = kd_val
        self.integral = 0
        self.prev_error = 0
        self.max_output = max_output_val

    
    def update(self, error, dt):
        # self.integral += error * dt
        self.integral += self.ki * 0.5 * (error + self.prev_error)*dt # Trapezoidal Integration
        # Prevent integral windup by limiting the integral term
        self.integral = max(min(self.integral, self.max_output), -self.max_output)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        pid_output = self.kp * error + self.integral + self.kd * derivative
    
        # Limit the PID output
        pid_output = max(min(pid_output, self.max_output), -self.max_output)

        self.prev_error = error
        return pid_output

# Define waypoints in mission (x, y, z in meters)
waypoints = [
    airsim.Vector3r(0, 0, -10),
    airsim.Vector3r(10, 0, -10),
    airsim.Vector3r(10, 10, -10),
    airsim.Vector3r(0, 10, -10),
    airsim.Vector3r(0, 0, -10)
]

# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a new directory for this run
results_dir = f'results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)

csv_file_name = 'simulation_results.csv'
# lidar_data_dir = "lidar_data"
# os.makedirs(lidar_data_dir, exist_ok=True)

# Initialize PID controllers for X, Y axes
pid_x = PIDController(kp_val=0.5, ki_val=0, kd_val=0,
                  max_output_val=5)
pid_y = PIDController(kp_val=0.5, ki_val=0, kd_val=0,
                  max_output_val=5)
pid_yaw = PIDController(kp_val=2.5, ki_val=0.0, kd_val=0.0,
                  max_output_val=3.14159265359)

# Define noise standard deviations for different simulations
# pos_noise_std = 0.0
yaw_noise_std = [0.0, 0.01, 0.05, 0.1, 0.3]
results = []

for i, std in enumerate(yaw_noise_std):
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
            # Convert yaw to degrees and print
            yaw_degrees = math.degrees(yaw)
            print(f"Yaw (degrees): {yaw_degrees}")

            #  # Add Gaussian noise to drone's current position data
            # noisy_position = add_noise(position, mean=0.0, std_dev=pos_noise_std)  # Adjust mean and std_dev as needed

            # Add Gaussian noise to yaw
            noisy_yaw = multiplicative_noise(yaw, mean=1.0, std_dev=std)
      

            # Calculate error in X-Y plane, although not require to control vx,vy, but to use yaw_mode API here,
            # must bring in vx,vy
            error_x = waypoint.x_val - position.x_val
            error_y = waypoint.y_val - position.y_val
            print(position.x_val, position.y_val)
            
              # Record position and time
            flight_path.append(( now-start_time, position))

            # Calculate distance traveled since last position
            distance = np.linalg.norm(position.to_numpy_array() - prev_position)
            total_distance += distance
            prev_position = position.to_numpy_array()


            # Update PID controllers
            control_x = pid_x.update(error_x, dt)
            control_y = pid_y.update(error_y, dt)

            # use PID Control logic on angle level and moving to the next waypoint asynchronously
            # Calculate desired yaw
            desired_yaw = math.atan2(waypoint.y_val - position.y_val, waypoint.x_val - position.x_val)

            # Update PID controllers
            # Calculate errors on yaw
            error_yaw = desired_yaw - noisy_yaw
            
            control_yaw = pid_yaw.update(error_yaw, dt)
            print (control_yaw)
        # regulate the velocity in X,Y axis, To BE completed!
            # max(min_value, min(val, max_value))
            client.rotateToYawAsync(yaw_degrees).join() 
            time.sleep(0.1) 
            client.moveByVelocityZAsync(vx=control_x, vy=control_y, z=waypoint.z_val, duration=dt, drivetrain=1, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(control_yaw)))


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
    client.reset()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    total_time = time.time() - start_time
    average_waypoint_distance = sum(waypoint_distances) / len(waypoint_distances) if waypoint_distances else 0

    results.append({'Noise std': std,
        'Total Distance Traveled (m)': total_distance,
        'Total Flight Time (s)': total_time,
        'Collision Count': collision_count,
        'Average Distance from Waypoints (m)': average_waypoint_distance,
        'PID X kp_val': pid_x.kp,  # Save kp_val for PID X
        'PID X ki_val': pid_x.ki,  # Save ki_val for PID X
        'PID X kd_val': pid_x.kd,  # Save kd_val for PID X
        'PID X max_output_val': pid_x.max_output,  # Save max_output_val for PID X
        'PID Y kp_val': pid_y.kp,  # Save kp_val for PID Y
        'PID Y ki_val': pid_y.ki,  # Save ki_val for PID Y
        'PID Y kd_val': pid_y.kd,  # Save kd_val for PID Y
        'PID Y max_output_val': pid_y.max_output,  # Save max_output_val for PID Y
        'PID Yaw kp_val': pid_yaw.kp,  # Save kp_val for PID X
        'PID Yaw ki_val': pid_yaw.ki,  # Save ki_val for PID X
        'PID Yaw kd_val': pid_yaw.kd,  # Save kd_val for PID X
        'PID Yaw max_output_val': pid_yaw.max_output  # Save max_output_val for PID X
        })
    
    # Extracting X, Y, Z coordinates
    times = [timestamp for timestamp, _ in flight_path]
    x_vals = [pos.x_val for _, pos  in flight_path]
    y_vals = [pos.y_val for _, pos  in flight_path]
    z_vals = [-pos.z_val for _, pos  in flight_path]

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
    ax.set_title(f'3D Flight Path Visualization with yaw Noise std {std}')
    plt.savefig(os.path.join(results_dir, f'flight_path_simulation_{i+1}.png'))
    plt.clf()

    plt.figure(figsize=(12, 10))

    # X Position vs Time
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
    plt.plot(times, x_vals, label='X Position')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position vs. Time')
    plt.legend()
    
    # Y Position vs Time
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
    plt.plot(times, y_vals, label='Y Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position vs. Time')
    plt.legend()

    # Z Position vs Time (Altitude)
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
    plt.plot(times, z_vals, label='Altitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs. Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'XYZ_vs_Time_{i+1}.png'))


df = pd.DataFrame(results)
df.to_excel(os.path.join(results_dir,'simulation_results.xlsx'), index=False)

# Print recorded flight path
# print("Flight path:")
# for pos, orient in flight_path:
#     print(f"Position: ({pos.x_val}, {pos.y_val}, {pos.z_val}), Orientation (quaternion): ({orient.x_val}, {orient.y_val}, {orient.z_val}, {orient.w_val})")


# Plotting the flight path
# ax.plot(x_vals, y_vals, z_vals, marker='o')


