import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to check proximity between current position and target waypoint
def is_close(current, target, threshold=1.0):
    return np.linalg.norm(np.array([current.x_val - target.x_val, current.y_val - target.y_val, current.z_val - target.z_val])) < threshold

# Function to add Gaussian noise to LiDAR data
def add_noise(lidar_points, mean=0.0, std_dev=1.0):
    noise = np.random.normal(mean, std_dev, lidar_points.shape)
    return lidar_points + noise

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Define waypoints
waypoints = [
    airsim.Vector3r(0, 0, -10),
    airsim.Vector3r(10, 0, -10),
    airsim.Vector3r(10, 10, -10),
    airsim.Vector3r(0, 10, -10),
    airsim.Vector3r(0, 0, -10)
]

# Takeoff
client.takeoffAsync().join()

flight_path = []

for waypoint in waypoints:
    # Move to the next waypoint asynchronously
    client.moveToPositionAsync(waypoint.x_val, waypoint.y_val, waypoint.z_val, 5)

    # Continuously sample state until the drone is close to the waypoint
    while not is_close(client.simGetVehiclePose().position, waypoint):
        # Retrieve LiDAR data
        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) < 3:
            continue
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        
        # Add Gaussian noise to LiDAR data
        noisy_points = add_noise(points, mean=0.0, std_dev=0.1)  # Adjust mean and std_dev as needed

        # Sample state
        state = client.simGetVehiclePose()
        position = state.position
        orientation = state.orientation  # Quaternion

        # Record position and orientation
        flight_path.append((position, orientation))

        # Sleep for a short duration to avoid excessive sampling
        time.sleep(0.1)

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
ax.plot(x_vals, y_vals, z_vals, marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Flight Path Visualization')

plt.show()

    