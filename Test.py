import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from global_path_planning_3d import RRTStar, Node

def add_obstacles(env, pos, radius):
    sphere_obst_dict = {
        "type": "sphere",
        'movable': False,
        "geometry": {"position": pos, "radius": radius},
    }
    sphere_obst = SphereObstacle(name=f'obstacle_{pos[0]}_{pos[1]}_{pos[2]}', content_dict=sphere_obst_dict)
    env.add_obstacle(sphere_obst)

def get_robot_config(ob):
    x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
    y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
    angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]
    return [x, y, angular]

class PIDcontroller:
    # a pid controller to control the angular velocity of the robot, aiming at the next target
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.error_sum = 0
        self.error_prev = 0

    def update(self, error, dt):
        self.error_sum += error * dt
        error_diff = (error - self.error_prev) / dt
        self.error_prev = error
        return self.kp * error + self.ki * self.error_sum + self.kd * error_diff
    

def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 0.5*np.pi,
            facing_direction = 'x',
        ),
    ]

    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )

    add_obstacles(env, [1.5, 0, 0], 0.5)
    add_obstacles(env, [1.5, 2, 3], 0.5)
    # # print the obstacles within the environment
    # for obstacle in env.get_obstacles():
    #     print(f"Obstacles in the environment: {env.get_obstacles()[obstacle].__dict__}")
    #     print(f"Obstacle position: {(env.get_obstacles()[obstacle])._content_dict['geometry']}")

    # bounds = {'xmin': 0, 'xmax': 7, 'ymin': 0, 'ymax': 7, 'zmin': 0, 'zmax': 2}
    # start = [0,0,0]
    # goal = [5,5,0]
    # obstacles = {i:(env.get_obstacles()[obstacle])._content_dict['geometry'] for i, obstacle in enumerate(env.get_obstacles())}
    # print(obstacles)
    # rrt_star_base = RRTStar(start, goal, bounds, obstacles, max_iter=100000)
    # path_points = rrt_star_base.full_run()

    # path_points = [rrt_star_base.nodes[node].position for node in path_points]
    # path_points.append([3,2,0])
    # for point in path_points:
    #     rounded_points = np.round(point, 2).astype(float)
    #     print(f'rounded_points: {rounded_points}')
    #     if rounded_points[1] != 0:
    #         x = float(rounded_points[0])
    #         y = float(rounded_points[1])
    #         z = float(rounded_points[2])
    #         add_obstacles(env, [x,y,z+2], 0.1)
    action = np.zeros(env.n())
    action[0] = 0 # forward velocity
    action[1] = 0 # angular velocity
    action[2] = 0 # joint 1
    action[3] = 0 # joint 2
    action[4] = 0 # joint 3
    action[5] = 0 # joint 4
    action[6] = 0 # joint 5
    action[7] = 0 # joint 6
    action[8] = 0 # joint 7
    action[9] = 0 # gripper -> snap deze nog niet helemaal
    action[10] = 0 # gripper -> snap deze nog niet helemaal

    # initial position
    ob = env.reset(
        pos=np.array([0, 0, -0.5*np.pi, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    ob = ob[0]
    #loop door de steps heen, voer een actie uit met env.step(action)
    for _ in range(n_steps):
        ob, *_ = env.step(action)

        # get the robot joint angles 
        print(ob['robot_0']['joint_state']['position'][3])
        # x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
        # y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
        # z = np.round(ob['robot_0']['joint_state']['position'], 1)[2]

        # print(f'x: {x}, y: {y}, z: {z}')
        # q_1_7 = inverse_kinematics(x, y, z)
        
        action[0] = -0.3
        action[3] = -0.5 # joint 1
        ob, *_ = env.step(action)
        


    env.close()

from sympy import symbols, atan2, sqrt, cos, sin

def inverse_kinematics(x, y, z):
    # Define joint variables
    q1, q2, q3, q4, q5, q6, q7 = symbols('q1 q2 q3 q4 q5 q6 q7')

    # DH table of the Panda arm
    DH_table = np.array([
        [0,       0,      0.333,  q1],
        [0,  -np.pi/2,      0,      q2],
        [0,   np.pi,      0.316,   q3],
        [0.0825, np.pi/2,   0,      q4],
        [-0.0825,-np.pi/2,  0.384,   q5],
        [0,   np.pi/2,      0,      q6],
        [0.088, np.pi/2,    0,      q7]
    ])

    # Initialize transformation matrix
    T = np.eye(4)

    # Calculate the end-effector position
    end_effector_pos = np.array([x, y, z, 1])

    # Iterate through each row in DH table
    for i in range(len(DH_table)):
        # Extract DH parameters
        a, alpha, d, theta = DH_table[i]

        # Homogeneous transformation matrix for the current joint
        A_i = np.array([
            [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
            [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

        # Update overall transformation matrix
        T = np.dot(T, A_i)

    # Extract the position of the end-effector from the transformation matrix
    end_effector_pos_calculated = T[:3, 3]

    # Calculate the joint angles using trigonometry and arctan
    q1_val = atan2(end_effector_pos_calculated[1], end_effector_pos_calculated[0])
    q3_val = sqrt(end_effector_pos_calculated[0]**2 + end_effector_pos_calculated[1]**2) - 0.333
    q2_val = atan2(end_effector_pos_calculated[2], q3_val)
    q4_val = q4  # Use the original value from the DH table
    q5_val = q5  # Use the original value from the DH table
    q6_val = q6  # Use the original value from the DH table
    q7_val = q7  # Use the original value from the DH table

    return q1_val, q2_val, q3_val, q4_val, q5_val, q6_val, q7_val


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)