import airsim
import numpy as np
import time
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from rotations import skew_symmetric, Quaternion

# Function to check proximity between current position and target waypoint
def is_close(current, target, threshold=0.1):
    return np.linalg.norm(np.array([current.x_val - target.x_val, current.y_val - target.y_val, current.z_val - target.z_val])) < threshold


class PIDController:
    def __init__(self, kp_x=1, ki_x=0, kd_x=10, max_output_x=1, kp_y=1, ki_y=0, kd_y=10, max_output_y=1):
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

    def update_controls(self, error_x, error_y, dt):
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

        # Update previous errors
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        return control_x, control_y


# Define waypoints in mission (x, y, z in meters)
waypoints = [
    airsim.Vector3r(0, 0, -10),
    airsim.Vector3r(10, 0, -10),
    airsim.Vector3r(10, 10, -10),
    airsim.Vector3r(0, 10, -10),
    airsim.Vector3r(0, 0, -10)
]

settings_file_path = ".\\settings.json"

# Load the current settings
with open(settings_file_path, 'r') as file:
    settings = json.load(file)
    

# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a new directory for this run
results_dir = f'results_{timestamp}'
os.makedirs(results_dir)

results = []
sensor_data = []

# Initialize PID controllers for X and Y axes with separate parameters
pid_controller = PIDController(kp_x=0.5, ki_x=0, kd_x=0.5, max_output_x=10,
                            kp_y=0.5, ki_y=0, kd_y=0.5, max_output_y=10) 

# #### EKF Constants ##############################################################################

# ################################################################################################
# # Now that our data is set up, we can start getting things ready for our solver. One of the
# # most important aspects of a filter is setting the estimated sensor variances correctly.
# # We set the values here.
# ################################################################################################
var_imu_f = 0.001
var_imu_w = 0.001

# ################################################################################################
# # We can also set up some constants that won't change for any iteration of our solver.
# ################################################################################################
g = np.array([0, 0, 9.81])  # gravity , for NED coordinates in AirSim
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian


# Run the AirSim simulation with the modified JSON settings

flight_path = []
# Initialize a list to store position errors
position_errors = []
err3 = []

total_distance = 0
collision_count = 0
start_time = time.time()

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

# Get current state as position and orientation
state = client.getMultirotorState() # only call state variable here once for initial use

# # Set initial values.
p_est = state.kinematics_estimated.position.to_numpy_array() # initial position estimates
v_est = state.kinematics_estimated.linear_velocity.to_numpy_array() # initial velocity estimates
orientation = state.kinematics_estimated.orientation 
q_est = Quaternion(euler=np.array([orientation.x_val, orientation.y_val, orientation.z_val])).to_numpy() ### gt.r[0] unit is rad, 4*1 np.array
imu_w = state.kinematics_estimated.angular_velocity.to_numpy_array() # initial Angular Velocity estimates
imu_f = state.kinematics_estimated.linear_acceleration.to_numpy_array() # initial linear acceleration estimates
p_cov = np.zeros(9)  # covariance of estimate, 9*9 matrix

for _, waypoint in enumerate(waypoints):

    last_time = time.time()

    # Continuously sample state until the drone is close to the waypoint
    while not is_close(client.simGetVehiclePose().position, waypoint):

        now = time.time()
        dt = now - last_time
        last_time = now


        # #### Use EKF filter to estimate current position #######################################################################

        # ################################################################################################
        # # Now that everything is set up, we can start taking in the IMU and GPS sensor data and creating estimates
        # # for our state
        # ################################################################################################
        # 1. Update state with IMU inputs

        # orientation = [imu_data.orientation.x_val, imu_data.orientation.y_val, imu_data.orientation.z_val]
        q_prev = Quaternion(*q_est) # previous orientation as a quaternion object

        q_curr = Quaternion(axis_angle=(imu_w*dt)) # current IMU orientation
        c_ns = q_prev.to_mat() # previous orientation as a matrix, 3*3 matrix

        
        f_ns = (c_ns @ imu_f) + g # calculate sum of forces, g needs to be +9.81
        # use Newton law to update position
        p_check = p_est + dt*v_est + 0.5*(dt**2)*f_ns
        print(p_check)

        # get position by calling state
        cur_position = client.simGetVehiclePose().position.to_numpy_array()
        # print(cur_position)

        # compare them 
        err3.append(p_check-cur_position)

        v_check = v_est + dt*f_ns
        q_check = q_prev.quat_mult_left(q_curr)

        # 1.1 Linearize the motion model and compute Jacobians
        f_jac = np.eye(9) # motion model jacobian with respect to last state, 9x9 identity matrix
        f_jac[0:3, 3:6] = np.eye(3)*dt # modifies the block from rows 0 to 2 and columns 3 to 5 of the f_jac matrix, part of the Jacobian represents the partial derivatives of the position states with respect to velocity states, 
        # assuming a basic kinematic model where position p is updated as p = p + v * dt
        f_jac[3:6, 6:9] = -skew_symmetric(c_ns @ imu_f.reshape(3,1))*dt # specifically for the velocity updates

        # 2. Propagate uncertainty
        q_cov = np.zeros((6, 6)) # IMU noise covariance
        q_cov[0:3, 0:3] = dt**2 * np.eye(3)*var_imu_f
        q_cov[3:6, 3:6] = dt**2 * np.eye(3)*var_imu_w
        p_cov_check = f_jac @ p_cov @ f_jac.T + l_jac @ q_cov @ l_jac.T

        # Update states (save)
        p_est = p_check
        v_est = v_check
        q_est = q_check
        p_cov = p_cov_check

        # Fetch IMU sensors data
        imu_data = client.getImuData()
        imu_f = imu_data.linear_acceleration.to_numpy_array()
        imu_w = imu_data.angular_velocity.to_numpy_array()
       

        # Record IMU, Barometer, GPS sensor data
        data_entry = {
            'Time(s)': now - start_time,
            'Angular Velocity X(rad/s)': imu_data.angular_velocity.x_val,
            'Angular Velocity Y(rad/s)': imu_data.angular_velocity.y_val,
            'Angular Velocity Z(rad/s)': imu_data.angular_velocity.z_val,
            'Linear Acceleration X(ms^-2)': imu_data.linear_acceleration.x_val,
            'Linear Acceleration Y(ms^-2)': imu_data.linear_acceleration.y_val,
            'Linear Acceleration Z(ms^-2)': imu_data.linear_acceleration.z_val,
            'Orientation W': imu_data.orientation.w_val,
            'Orientation X': imu_data.orientation.x_val,
            'Orientation Y': imu_data.orientation.y_val,
            'Orientation Z': imu_data.orientation.z_val
         
        }

        sensor_data.append(data_entry)

        # Record position and time
        flight_path.append(( now-start_time, p_check))
        position_errors.append((waypoint.x_val - p_check[0], waypoint.y_val - p_check[1], waypoint.z_val + p_check[2]))


        # use PID Control logic moving to the next waypoint asynchronously
        # Calculate position error in X-Y plane
        error_x = waypoint.x_val - p_check[0]
        error_y = waypoint.y_val - p_check[1]

        # Update PID controls for X and Y simultaneously
        control_x, control_y = pid_controller.update_controls(error_x, error_y, dt)

        # Apply controls
        client.moveByVelocityZAsync(vx=control_x, vy=control_y, z=waypoint.z_val, duration=dt)

        # Check for collision
        collision_info = client.simGetCollisionInfo()

        if collision_info.has_collided:
            print("Collision detected!")
            collision_count += 1
        
        # Sleep to avoid excessive sampling
        # time.sleep(0.1)

# Land
client.reset()
client.armDisarm(False)
client.enableApiControl(False)

total_time = time.time() - start_time

results.append({
    # 'Total Distance Traveled (m)': total_distance,
    'Total Flight Time (s)': total_time,
    'Collision Count': collision_count,
    'PID X kp_val': pid_controller.kp_x,  
    'PID X ki_val': pid_controller.ki_x,  
    'PID X kd_val': pid_controller.kd_x,  
    'PID X max_output_val': pid_controller.max_output_x,  
    'PID Y kp_val': pid_controller.kp_y,  
    'PID Y ki_val': pid_controller.ki_y,  
    'PID Y kd_val': pid_controller.kd_y,  
    'PID Y max_output_val': pid_controller.max_output_y,
    "AngularRandomWalk": settings['Vehicles']['Drone1']["Sensors"]["Imu"]['AngularRandomWalk'],
    "GyroBiasStabilityTau": settings['Vehicles']['Drone1']["Sensors"]["Imu"]['GyroBiasStabilityTau'],
    "GyroBiasStability": settings['Vehicles']['Drone1']["Sensors"]["Imu"]['GyroBiasStability'],
    "VelocityRandomWalk": settings['Vehicles']['Drone1']["Sensors"]["Imu"]['VelocityRandomWalk'],
    "AccelBiasStabilityTau": settings['Vehicles']['Drone1']["Sensors"]["Imu"]['AccelBiasStabilityTau'],
    "AccelBiasStability": settings['Vehicles']['Drone1']["Sensors"]["Imu"]['AccelBiasStability'],
    "EphTimeConstant": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['EphTimeConstant'],
    "EpvTimeConstant": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['EpvTimeConstant'],
    "EphInitial": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['EphInitial'],
    "EpvInitial": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['EpvInitial'],
    "EphFinal": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['EphFinal'],
    "EpvFinal": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['EpvFinal'],
    "EphMin3d": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['EphMin3d'],
    "EphMin2d": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['EphMin2d'],
    "UpdateLatency": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['UpdateLatency'],
    "UpdateFrequency": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['UpdateFrequency'],
    "StartupDelay": settings['Vehicles']['Drone1']["Sensors"]["Gps"]['StartupDelay']
    
})

# Extracting X, Y, Z coordinates
times = [timestamp for timestamp, _ in flight_path]
x_vals = [pos[0] for _, pos in flight_path]
y_vals = [pos[1]  for _, pos in flight_path]
z_vals = [pos[2]  for _, pos in flight_path] #  Negate Z to show altitude above ground

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
ax.set_title(f'3D Flight Path Visualization')
plt.savefig(os.path.join(results_dir, 'flight_path_simulation.png'))
plt.clf()

plt.figure(figsize=(12, 10))

# X Position vs Time
plt.subplot(3, 1, 1)  
plt.plot(times, x_vals, label='X Position')
plt.xlabel('Time (s)')
plt.ylabel('X Position (m)')
plt.title('X Position vs. Time')
plt.legend()

# Y Position vs Time
plt.subplot(3, 1, 2)  
plt.plot(times, y_vals, label='Y Position')
plt.xlabel('Time (s)')
plt.ylabel('Y Position (m)')
plt.title('Y Position vs. Time')
plt.legend()

# Z Position vs Time (Altitude)
plt.subplot(3, 1, 3)  
plt.plot(times, z_vals, label='Altitude')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs. Time')
plt.legend()
plt.savefig(os.path.join(results_dir, 'XYZ_vs_Time.png'))


errors = np.array(position_errors)
# Plot the errors over time
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(times, errors[:, 0], label='Error in X')
plt.xlabel('Time(s)')
plt.ylabel('Error')
plt.title('Position Error in X')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(times, errors[:, 1], label='Error in Y')
plt.xlabel('Time(s)')
plt.ylabel('Error')
plt.title('Position Error in Y')
plt.legend()

# Z Position vs Time (Altitude)
plt.subplot(3, 1, 3)  
plt.plot(times, errors[:, 2], label='Error in Altitude')
plt.xlabel('Time(s)')
plt.ylabel('Error')
plt.title('Error in Altitude')
plt.legend()
plt.savefig(os.path.join(results_dir, 'Error_vs_Time.png'))


plt.figure(figsize=(10, 6))
X_diff = [item[0] for item in err3]
Y_diff = [item[1] for item in err3]
plt.plot(times, X_diff, label='X Difference')
plt.plot(times, Y_diff, label='Y Difference')
# plt.plot(times, err3[:, 2], label='Z Difference')
plt.title('Newton law estimtes vs AirSim State Position Differences Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position Difference (meters)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'error3_only_IMU_NoGPS.png'))


df = pd.DataFrame(results)
df.to_excel(os.path.join(results_dir,'simulation_results.xlsx'), index=False)

sensor_data_df = pd.DataFrame(sensor_data)
sensor_data_df.to_excel(os.path.join(results_dir,'sensor_data.xlsx'), index=False)
