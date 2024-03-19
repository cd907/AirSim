import airsim
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to check proximity between current position and target waypoint
def is_close(current, target, threshold=1.0):
    return np.linalg.norm(np.array([current.x_val - target.x_val, current.y_val - target.y_val, current.z_val - target.z_val])) < threshold

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

flight_path = []

lidar_data_dir = "lidar_data"
os.makedirs(lidar_data_dir, exist_ok=True)

# Initialize PID controllers for X, Y, Z axes
pid_x = PIDController()
pid_y = PIDController()

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()


for idx, waypoint in enumerate(waypoints):

    last_time = time.time()
    # Continuously sample state until the drone is close to the waypoint
    while not is_close(client.simGetVehiclePose().position, waypoint):

        now = time.time()
        dt = now - last_time
        last_time = now

        # Get current position
        state = client.simGetVehiclePose()
        position = state.position
        orientation = state.orientation  # Quaternion


        # Record position and orientation
        flight_path.append((position, orientation))

        # use PID Control logic moving to the next waypoint asynchronously
        # Calculate position error in X-Y plane
        error_x = waypoint.x_val - position.x_val
        error_y = waypoint.y_val - position.y_val

         # Update PID controllers
        control_x = pid_x.update(error_x, dt)
        control_y = pid_y.update(error_y, dt)
       
       # regulate the velocity in X,Y axis, To BE completed!
        # max(min_value, min(val, max_value))

        client.moveByVelocityZAsync(vx=control_x, vy=control_y, z=waypoint.z_val, duration=1)

        # Check for collision
        collision_info = client.simGetCollisionInfo()

        if collision_info.has_collided:
            print("Collision detected!")
        else:
            print("No collision detected.")
        
        # Sleep for a short duration to avoid excessive sampling
        time.sleep(0.1)

    # Retrieve LiDAR data at current position
    lidar_data = client.getLidarData()
    if len(lidar_data.point_cloud) < 3:
        continue
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    
    # Save LiDAR data to a file
    lidar_filename = os.path.join(lidar_data_dir, f"waypoint_{idx+1}_lidar_data.csv")
    np.savetxt(lidar_filename, points, delimiter=",", fmt='%f', header='x,y,z', comments='')
    
    print(f"Reached waypoint {idx+1}, LiDAR data saved to {lidar_filename}")

# Land
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

# Print recorded flight path
print("Flight path:")
for pos, orient in flight_path:
    print(f"Position: ({pos.x_val}, {pos.y_val}, {pos.z_val}), Orientation (quaternion): ({orient.x_val}, {orient.y_val}, {orient.z_val}, {orient.w_val})")


# Extracting X, Y, Z coordinates
x_vals = [pos.x_val for pos, _ in flight_path]
y_vals = [pos.y_val for pos, _ in flight_path]
z_vals = [pos.z_val for pos, _ in flight_path]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the flight path
# ax.plot(x_vals, y_vals, z_vals, marker='o')

# Plotting the flight path as a scatter plot
ax.scatter(x_vals, y_vals, z_vals, c='blue', s=10) 

# Highlight the start point in green
ax.scatter(x_vals[0], y_vals[0], z_vals[0], c='green', s=50, label='Start Point')

# Highlight the end point in red
ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], c='red', s=50, label='End Point')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Flight Path Visualization')

plt.show()

# Plotting the LiDAR data
df_no_noise = pd.read_csv(lidar_filename)
fig_no_noise = plt.figure()
ax_no_noise = fig_no_noise.add_subplot(111, projection='3d')
ax_no_noise.scatter(df_no_noise['x'], df_no_noise['y'], df_no_noise['z'])
ax_no_noise.set_title('No Noise LiDAR Data')
ax_no_noise.set_xlabel('X axis')
ax_no_noise.set_ylabel('Y axis')
ax_no_noise.set_zlabel('Z axis')

plt.show()