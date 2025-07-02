import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
from scipy.spatial.transform import Rotation as R
import heapq
from enum import Enum

from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseArray, TransformStamped, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from scipy.ndimage import grey_dilation
import math


class State(Enum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    AVOIDING_OBSTACLE = 3


class AmclNode(Node):
    def __init__(self):
        super().__init__('my_py_amcl')

        # --- Parameters ---
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('initial_pose_topic', 'initialpose')
        self.declare_parameter('laser_max_range', 3.5)
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('obstacle_detection_distance', 0.3)
        self.declare_parameter('obstacle_avoidance_turn_speed', 1.5)

        # --- Parameters to set ---
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('alpha1', 0.1)
        self.declare_parameter('alpha2', 0.1)
        self.declare_parameter('alpha3', 0.2)
        self.declare_parameter('alpha4', 0.2)
        self.declare_parameter('z_hit', 0.9)
        self.declare_parameter('z_rand', 0.1)
        self.declare_parameter('lookahead_distance', 0.3)
        self.declare_parameter('linear_velocity', 0.2)
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('path_pruning_distance', 0.3)
        self.declare_parameter('safety_margin_cells', 4)
        self.front_angle_deg = 20
        self.avoidance_turn_direction = None
        self.obstacle_advancing = False
        self.obstacle_advance_start_time = None

        self.num_particles = self.get_parameter('num_particles').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.laser_max_range = self.get_parameter('laser_max_range').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value
        self.alphas = np.array([
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ])
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_pruning_distance = self.get_parameter('path_pruning_distance').value
        self.safety_margin_cells = self.get_parameter('safety_margin_cells').value
        self.obstacle_detection_distance = self.get_parameter('obstacle_detection_distance').value
        self.obstacle_avoidance_turn_speed = self.get_parameter('obstacle_avoidance_turn_speed').value

        # --- State ---
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.map_data = None
        self.latest_scan = None
        self.initial_pose_received = False
        self.map_received = False
        self.last_odom_pose = None
        self.state = State.IDLE
        self.current_path = None
        self.goal_pose = None
        self.inflated_grid = None
        self.obstacle_grid = None
        self.obstacle_avoidance_start_yaw = None
        self.obstacle_avoidance_last_yaw = None
        self.obstacle_avoidance_cumulative_angle = 0.0
        self.obstacle_avoidance_active = False

        # --- ROS 2 Interfaces ---
        map_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        scan_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.map_sub = self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, scan_qos)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.get_parameter('initial_pose_topic').value, self.initial_pose_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'amcl_pose', 10)
        self.particle_pub = self.create_publisher(MarkerArray, 'particle_cloud', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('MyPyAMCL node initialized.')

    def inflate_map(self):
        """
        Inflate the obstacle map to add a safety margin around occupied cells.
        """
        structuring_element = np.ones((2 * self.safety_margin_cells + 1, 2 * self.safety_margin_cells + 1), dtype=np.uint8)
        binary_map = (self.grid > 50).astype(np.uint8)  

        inflated = grey_dilation(binary_map, footprint=structuring_element)
        self.inflated_grid = inflated

    def map_callback(self, msg):
        """
        Callback to handle the occupancy grid map when received.
        Stores the map, reshapes it, and inflates it to account for safety margin.
        """
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            self.grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
            self.inflate_map()
            self.get_logger().info('Map and inflated map processed.')

    def scan_callback(self, msg):
        """
        Callback to store the most recent LIDAR scan.
        """
        self.latest_scan = msg

    def goal_callback(self, msg):
        """
        Callback to receive and process a navigation goal.
        Switches to planning state once the goal pose is set.
        """
        if self.map_data is None:
            self.get_logger().warn("Goal received, but map is not available yet. Ignoring goal.")
            return

        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Goal received in frame '{msg.header.frame_id}', but expected '{self.map_frame_id}'. Ignoring.")
            return

        self.goal_pose = msg.pose
        self.get_logger().info(f"New goal received: ({self.goal_pose.position.x:.2f}, {self.goal_pose.position.y:.2f}). State -> PLANNING")
        self.initial_pose_sub
        self.stop_robot()
        self.state = State.PLANNING
        self.current_path = None

    def initial_pose_callback(self, msg):
        """
        Callback to set the initial pose if the frame is correct.
        """
        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Initial pose frame is '{msg.header.frame_id}' but expected '{self.map_frame_id}'. Ignoring.")
            return
        self.get_logger().info('Initial pose received.')
        self.initialize_particles(msg.pose.pose)
        self.initial_pose_received = True
        self.last_odom_pose = None 

    def initialize_particles(self, initial_pose, max_attempts=100):
        """
        Initialize particles for particle filter around a given initial pose with Gaussian noise.
        """
        x0 = initial_pose.position.x
        y0 = initial_pose.position.y
        orientation_q = initial_pose.orientation
        theta0 = R.from_quat([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]).as_euler('xyz')[2]

        for i in range(self.num_particles):
            for _ in range(max_attempts):
                x = x0 + np.random.normal(0, 0.2)
                y = y0 + np.random.normal(0, 0.2)
                theta = self.normalize_angle(theta0 + np.random.normal(0, 0.1))

                gx, gy = self.world_to_grid(x, y)

                if 0 <= gx < self.map_data.info.width and 0 <= gy < self.map_data.info.height:
                    if self.inflated_grid[gy, gx] == 0:
                        break
            self.particles[i] = np.array([x, y, theta])
  
        self.publish_particles()

    def initialize_particles_randomly(self):
        """
        Initialize particles randomly across free cells in the map.
        """
        for i in range(self.num_particles):
            x, y, theta = self.sample_random_point()

            self.particles[i] = np.array([x, y, theta])

        self.publish_particles()

    def sample_random_point(self, max_attempts=100):
        """
        Samples a random free point (x, y, theta) from the inflated map.
        Returns world coordinates in a free cell, or (0.0, 0.0, 0.0) if none found.
        """
        for _ in range(max_attempts):
            cell_x = np.random.randint(0, self.map_data.info.width)
            cell_y = np.random.randint(0, self.map_data.info.height)

            if self.inflated_grid[cell_y, cell_x] == 0:
                world_x = cell_x * self.map_data.info.resolution + self.map_data.info.origin.position.x
                world_y = cell_y * self.map_data.info.resolution + self.map_data.info.origin.position.y
                theta = self.normalize_angle(np.random.uniform(0, 2 * np.pi))
                return [world_x, world_y, theta]

        self.get_logger().warn("Failed to sample free point after max_attempts.")
        return [0.0, 0.0, 0.0]

    def normalize_weights(self):
        """
        Normalizes the particle weights to ensure they sum to 1.
        """
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def stop_robot(self):
        """
        Publishes a zero Twist message to stop all robot motion.
        """
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def timer_callback(self):
        """
        Main loop executed periodically to update localization and control the robot's behavior
        based on a state machine. Handles localization, path planning, navigation, and obstacle avoidance.
        """
        if not self.map_received:
            return

        # --- Localization (always running) ---
        if self.latest_scan is None:
            return

        if not self.initial_pose_received:
            self.initialize_particles_randomly()
            self.initial_pose_received = True
            return

        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            if self.state in [State.NAVIGATING, State.AVOIDING_OBSTACLE]:
                self.stop_robot()
            return


        self.motion_model(current_odom_tf) 
        self.measurement_model()
        self.normalize_weights()
        estimated_pose, estimated_theta = self.estimate_pose()
        self.resample()
        self.normalize_weights()

        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)
        
        # ---- State machine ----
        if self.state == State.PLANNING:
            if self.goal_pose is not None:
                self.current_path = self.plan_path_rrt(estimated_pose.position, self.goal_pose.position)

                if self.current_path:
                    self.smooth_path()
                    self.publish_path()
                    self.state = State.NAVIGATING
                    self.get_logger().info("Path planned. State -> NAVIGATING")
                else:
                    self.get_logger().warn("Failed to plan path. Staying in PLANNING.")
            else:
                self.get_logger().warn("No goal pose. Staying in PLANNING.")

        elif self.state == State.NAVIGATING:
            if self.detect_obstacle():
                self.stop_robot()
                self.get_logger().info("Obstacle detected. State -> AVOIDING_OBSTACLE")
                self.state = State.AVOIDING_OBSTACLE
                self.obstacle_avoidance_last_yaw = estimated_theta
                self.obstacle_avoidance_cumulative_angle = 0.0

            if self.reached_goal(estimated_pose.position):
                self.get_logger().info("Goal reached! State -> IDLE")
                self.state = State.IDLE
                self.stop_robot()
            else:
                self.follow_path(estimated_pose)

        elif self.state == State.AVOIDING_OBSTACLE:
            if self.avoid_obstacle(estimated_theta):
                self.stop_robot()
                self.get_logger().info("Obstacle cleared. State -> NAVIGATING")
                self.avoidance_turn_direction = None
                self.obstacle_avoidance_cumulative_angle = 0.0
                self.obstacle_avoidance_last_yaw = None
                self.state = State.NAVIGATING

        elif self.state == State.IDLE:
            self.stop_robot()  

    def get_odom_transform(self):
        """
        Get the latest transform from odometry frame to robot base frame.
        """
        try:
            return self.tf_buffer.lookup_transform(self.odom_frame_id, self.base_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'Could not get transform from {self.odom_frame_id} to {self.base_frame_id}. Skipping update. Error: {e}', throttle_duration_sec=2.0)
            return None

    def motion_model(self, current_odom_tf):
        """
        Applies a probabilistic motion model to propagate each particle from the previous 
        pose using the difference in odometry between the current and last one.
        """
        current_odom_pose = current_odom_tf.transform

        x = current_odom_pose.translation.x
        y = current_odom_pose.translation.y
        q_x = current_odom_pose.rotation.x
        q_y = current_odom_pose.rotation.y
        q_z = current_odom_pose.rotation.z
        q_w = current_odom_pose.rotation.w

        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2]

        if self.last_odom_pose is not None:
            prev_x, prev_y, prev_theta, prev_q = self.last_odom_pose
            prev_rotation = R.from_quat(prev_q)

            dx = x - prev_x
            dy = y - prev_y
            delta_t = np.sqrt(dx**2 + dy**2)

            relative_rotation = prev_rotation.inv() * current_rotation
            delta_r1 = relative_rotation.as_euler('xyz', degrees=False)[2] 
            delta_r2 = 0

            for i, p in enumerate(self.particles):
                sigma_delta_rot1 = self.alphas[0]
                delta_rot1_noisy = delta_r1 + np.random.normal(0,sigma_delta_rot1)

                sigma_translation = self.alphas[2]
                translation_noisy = delta_t + np.random.normal(0,sigma_translation)

                sigma_delta_rot2 = self.alphas[1]
                delta_rot2_noisy = delta_r2 + np.random.normal(0,sigma_delta_rot2)

                x_new = p[0] + translation_noisy * np.cos(p[2]+delta_rot1_noisy)
                y_new = p[1] + translation_noisy * np.sin(p[2]+delta_rot1_noisy)

                angle = self.normalize_angle(p[2] + delta_rot1_noisy + delta_rot2_noisy)
                self.particles[i] = np.array([x_new,y_new,angle])

        self.last_odom_pose = (x, y, theta, [q_x, q_y, q_z, q_w])


    def measurement_model(self):
        """
        Uses a simplified beam-based measurement model to update the weight of each 
        particle based on how well it explains the current LIDAR scan.
        """
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        map_img = np.array(self.map_data.data).reshape((map_h, map_w))

        angles = np.linspace(self.latest_scan.angle_min, self.latest_scan.angle_max, len(self.latest_scan.ranges))

        for i, p in enumerate(self.particles):
            x, y, theta = p
            weight = 1.0

            for j, angle in enumerate(angles[::15]):  # Use 1 out of every 15 beams to speed up the computation.
                scan_angle = theta + angle
                measured_range = self.latest_scan.ranges[j*15]

                if measured_range >= self.laser_max_range or np.isnan(measured_range):
                    continue

                expected_x = x + measured_range * np.cos(scan_angle)
                expected_y = y + measured_range * np.sin(scan_angle)

                gx, gy = self.world_to_grid(expected_x, expected_y)
                if 0 <= gx < map_w and 0 <= gy < map_h:
                    cell = map_img[gy, gx]
                    if cell > 0:
                        prob_hit = self.z_hit
                    else:
                        prob_hit = 0.01 
                else:
                    prob_hit = 0.01

                prob = prob_hit + self.z_rand
                weight *= prob 

            self.weights[i] = weight + 1e-300

    def resample(self):
        """
        Resamples the set of particles based on their weights using the 
        stochastic universal resampling algorithm.
        """
        cdf_sum=0
        p_cdf=[]

        for k in range(self.num_particles):
            cdf_sum = cdf_sum+self.weights[k]
            p_cdf.append(cdf_sum)

        step = 1.0/self.num_particles
        seed = np.random.uniform(0, step)

        p_sampled=[]
        w_sampled=[]
        last_index = 0
        for h in range(self.num_particles):
            while last_index < self.num_particles - 1 and seed > p_cdf[last_index]:
                last_index += 1
            p_sampled.append(np.copy(self.particles[last_index]))
            w_sampled.append(self.weights[last_index])
            seed = seed+step
        self.particles = np.array(p_sampled)
        self.weights = np.array(w_sampled) 

    def estimate_pose(self):
        """
        Estimates the robot's pose as a weighted average of the particles.
        """
        x_avg = 0
        y_avg = 0

        for i,p in enumerate(self.particles):
            weighted_p = p*self.weights[i]
            x_avg += weighted_p[0]
            y_avg += weighted_p[1]

        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        theta_avg = np.arctan2(sin_sum, cos_sum)

        estimated_pose = Pose()
        estimated_pose.position.x = x_avg
        estimated_pose.position.y = y_avg
        q = R.from_euler('z', theta_avg).as_quat()
        estimated_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return estimated_pose, theta_avg

    def collision_between(self, p1, p2):
        """
        Checks whether the straight line between two points intersects any obstacle in the occupancy grid.
        """
        distance = np.linalg.norm(np.array(p2) - np.array(p1))
        if distance < self.map_data.info.resolution:
            return False
        
        num_points = int(np.linalg.norm(p2 - p1) / self.map_data.info.resolution)
        for i in range(num_points + 1):
            wx = p1[0] + (p2[0] - p1[0]) * i / num_points
            wy = p1[1] + (p2[1] - p1[1]) * i / num_points

            gx, gy = self.world_to_grid(wx, wy)

            if 0 <= gx < self.map_data.info.width and 0 <= gy < self.map_data.info.height:
                if self.inflated_grid[gy, gx] > 0:
                    return True

        return False

    def plan_path_rrt(self, pose, goal, max_iters=2000, goal_sample_rate=0.1, step_size=0.4):
        """
        Plans a path from the start pose to the goal using the RRT algorithm.
        Returns the path as a list of points if successful, or None otherwise.
        """
        start = np.array([pose.x, pose.y])
        goal = np.array([goal.x, goal.y])
        tree = [{"pose": np.array(start), "parent": None}]

        for i in range(max_iters):
            if np.random.rand() < goal_sample_rate:
                sample = goal 
            else:
                sample = np.array(self.sample_random_point()[:2])

            nearest_index = min(range(len(tree)), key=lambda i: np.linalg.norm(tree[i]["pose"] - sample))
            
            direction = sample - tree[nearest_index]['pose']
            length = np.linalg.norm(direction)
            if length == 0:
                continue  
            direction = direction / length  
            possible_new_pose = tree[nearest_index]['pose'] + direction * step_size

            if not self.collision_between(tree[nearest_index]['pose'], possible_new_pose):
                tree.append({"pose": possible_new_pose, "parent": nearest_index})

                if np.linalg.norm(possible_new_pose - goal) < step_size:
                    if not self.collision_between(possible_new_pose, goal):
                        path = [goal]
                        idx = -1
                        while idx is not None:
                            path.insert(0, tree[idx]['pose'])
                            idx = tree[idx]['parent']
                        return path
                            
        return None
    
    def smooth_path(self, max_segment_length=0.1):
        """
        Simplifies and refines the current path by removing unnecessary intermediate points
        and inserting evenly spaced points between segments to ensure smooth navigation.
        """
        smoothed = [self.current_path[0]]
        i = 0
        while i < len(self.current_path) - 1:
            j = i+1
            while j<len(self.current_path) and not self.collision_between(np.array(self.current_path[i]), np.array(self.current_path[j])):
                j += 1

            i = j-1
            smoothed.append(self.current_path[i])

        refined = [smoothed[0]]
        for k in range(1, len(smoothed)):
            p1 = np.array(refined[-1])
            p2 = np.array(smoothed[k])
            distance = np.linalg.norm(p2 - p1)
            num_insertions = int(distance / max_segment_length)

            if num_insertions > 0:
                for n in range(1, num_insertions + 1):
                    alpha = n / (num_insertions + 1)
                    intermediate = (1 - alpha) * p1 + alpha * p2
                    refined.append(intermediate.tolist())

        refined.append(smoothed[-1])
        self.current_path = refined

    def reached_goal(self, pose):
        """
        Checks whether the robot has reached the goal position within a specified tolerance.
        """   
        goal =  self.goal_pose.position
        pose_arr = np.array([pose.x, pose.y])
        goal_arr = np.array([goal.x, goal.y])
        return np.linalg.norm(pose_arr - goal_arr) < self.goal_tolerance
    
    def pure_pursuit(self, pose):
        """
        Implements the Pure Pursuit algorithm to compute motion commands based on the robot's current pose.
        Returns linear and angular velocity.
        """
        x = pose.position.x
        y = pose.position.y
        q = pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        theta = R.from_quat(quat).as_euler('xyz')[2] 

        goal_point = None
        for px, py in self.current_path:
            dist = np.linalg.norm([px - x, py - y])
            if dist >= self.lookahead_distance:
                goal_point = (px, py)
                break

        if goal_point is None:
            goal_point = self.current_path[-1]
            
        dx = goal_point[0] - x
        dy = goal_point[1] - y
        local_x = math.cos(-theta) * dx - math.sin(-theta) * dy
        local_y = math.sin(-theta) * dx + math.cos(-theta) * dy

        angle_to_target = math.atan2(local_y, local_x)
        if abs(angle_to_target) > np.deg2rad(90):
            angular_velocity = 0.8 if angle_to_target > 0 else -0.8
            return 0.0, angular_velocity

        curvature = (2 * local_y) / (self.lookahead_distance ** 2)
        angular_velocity = self.linear_velocity * curvature

        # Adjust linear velocity based on curvature.
        max_linear = self.linear_velocity
        min_linear = 0.05 
        k = 0.5 
        linear_velocity = max(min_linear, max_linear / (1 + k * abs(curvature)))
        return linear_velocity, angular_velocity

    def follow_path(self, pose):
        """
        Executes path following using the Pure Pursuit algorithm,
        continuously prunes the path to remove points too close to the robot
        and publishes velocity commands to move the robot.
        """
        if not self.current_path or len(self.current_path) < 2:
            self.stop_robot()
            return

        robot_pos = np.array([pose.position.x, pose.position.y])
        if len(self.current_path) > 2:
            new_path = [pt for pt in self.current_path[:-1] if np.linalg.norm(robot_pos - np.array(pt)) > self.path_pruning_distance]
            new_path.append(self.current_path[-1])
            self.current_path = new_path

        if not self.current_path:
            self.stop_robot()
            return
        
        linear_vel, angular_vel = self.pure_pursuit(pose)

        twist = Twist()
        twist.linear.x = linear_vel
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)
    
    def detect_obstacle(self):
        """
        Checks if there is an obstacle in front of the robot.
        """
        front_angle_rad = np.deg2rad(self.front_angle_deg)
        scan = self.latest_scan
        ranges = scan.ranges
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        center_index = int((0 - angle_min) / angle_increment)
        delta_index = int(front_angle_rad / angle_increment)

        start_index = max(0, center_index - delta_index)
        end_index = min(len(ranges), center_index + delta_index + 1)

        for i in range(start_index, end_index):
            distance = ranges[i]
            if not math.isinf(distance) and not math.isnan(distance):
                if distance < self.obstacle_detection_distance:
                    return True 

        return False 
    
    def turn_direction(self):
        """
        Determines the preferred direction to turn to avoid an obstacle,
        based on comparing the amount of free space on the left and right sides.
        """
        if self.avoidance_turn_direction is not None:
            return self.avoidance_turn_direction
        
        ranges = np.array(self.latest_scan.ranges)
        valid_ranges = np.where((ranges > 0.1) & (ranges < self.laser_max_range), ranges, np.nan)

        fourth = len(valid_ranges) // 4
        half = len(valid_ranges) // 2

        right_indices = valid_ranges[fourth:half]     
        left_indices  = valid_ranges[half:half+fourth]    

        right_mean = np.nanmean(right_indices)
        left_mean = np.nanmean(left_indices)

        if np.isnan(left_mean): left_mean = 0
        if np.isnan(right_mean): right_mean = 0

        self.get_logger().info(f"left_mean: {left_mean:.2f}, right_mean: {right_mean:.2f}")
        if left_mean > right_mean:
            self.avoidance_turn_direction = 1
            return 1 
        else:
            self.avoidance_turn_direction = -1
            return -1  

    def avoid_obstacle(self, current_yaw):
        """
        Handles the robot's behavior to avoid obstacles when one is detected.
        """
        if self.obstacle_advancing:
            if (self.get_clock().now().nanoseconds / 1e9 ) - self.obstacle_advance_start_time  >= 1:
                self.obstacle_advancing = False
                self.obstacle_advance_start_time = None
                return True
            
            if not self.detect_obstacle():
                twist = Twist()
                twist.linear.x = 0.1
                self.cmd_vel_pub.publish(twist)   
            else:
                self.obstacle_advancing = False
                self.obstacle_advance_start_time = None
            return False
      
        delta_yaw = self.normalize_angle(current_yaw - self.obstacle_avoidance_last_yaw)
        self.obstacle_avoidance_cumulative_angle += abs(delta_yaw)
        self.obstacle_avoidance_last_yaw = current_yaw

        if self.obstacle_avoidance_cumulative_angle >= np.deg2rad(45) and not self.detect_obstacle():
            self.obstacle_advancing = True
            self.obstacle_advance_start_time = (self.get_clock().now().nanoseconds / 1e9)
            return False
        
        twist = Twist()
        twist.angular.z = self.obstacle_avoidance_turn_speed * self.turn_direction()
        self.cmd_vel_pub.publish(twist)
        return False
    
    def publish_pose(self, estimated_pose):
        """
        Publishes the estimated pose of the robot as a PoseWithCovarianceStamped message.
        """
        p = PoseWithCovarianceStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = self.map_frame_id
        p.pose.pose = estimated_pose
        self.pose_pub.publish(p)

    def publish_particles(self):
        """
        Publishes the current set of particles as a visualization marker array for RViz.
        """
        ma = MarkerArray()
        for i, p in enumerate(self.particles):
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            q = R.from_euler('z', p[2]).as_quat()
            marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 0.5
            marker.color.r = 1.0
            ma.markers.append(marker)
        self.particle_pub.publish(ma)

    def publish_transform(self, estimated_pose, odom_tf):
        """
        Publishes the transform between the map and odom frames based on the estimated pose.
        """
        map_to_base_mat = self.pose_to_matrix(estimated_pose)
        odom_to_base_mat = self.transform_to_matrix(odom_tf.transform)
        map_to_odom_mat = np.dot(map_to_base_mat, np.linalg.inv(odom_to_base_mat))

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame_id
        t.child_frame_id = self.odom_frame_id
        t.transform.translation.x = map_to_odom_mat[0, 3]
        t.transform.translation.y = map_to_odom_mat[1, 3]
        t.transform.translation.z = 0.0

        rot = R.from_matrix(map_to_odom_mat[:3, :3])
        q = rot.as_quat()
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        self.tf_broadcaster.sendTransform(t)

    def pose_to_matrix(self, pose):
        """
        Converts a Pose message into a 4x4 transformation matrix.
        """
        q = pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return mat

    def transform_to_matrix(self, transform):
        """
        Converts a Transform message into a 4x4 transformation matrix.
        """
        q = transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        t = transform.translation
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def world_to_grid(self, wx, wy):
        """
        Converts world coordinates (meters) to grid map coordinates (cells).
        """
        gx = int((wx - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        gy = int((wy - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        """
        Converts grid map coordinates (cells) to world coordinates (meters).
        """
        wx = gx * self.map_data.info.resolution + self.map_data.info.origin.position.x
        wy = gy * self.map_data.info.resolution + self.map_data.info.origin.position.y
        return (wx, wy)


    def publish_path(self):
        """
        Publishes the current path as a Path message.
        """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame_id

        for point in self.current_path:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = self.map_frame_id
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

    def normalize_angle(self, angle):
        """
        Normalizes an angle to the range [-pi, pi].
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = AmclNode() 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()