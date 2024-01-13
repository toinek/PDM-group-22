import warnings
import gymnasium as gym
import numpy as np

import pybullet as p
import pybullet_data
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


def get_joint_angles(ob):
    q1 = np.round(ob['robot_0']['joint_state']['position'], 1)[3] - np.pi/4
    q2 = np.round(ob['robot_0']['joint_state']['position'], 1)[4]
    q3 = np.round(ob['robot_0']['joint_state']['position'], 1)[5]
    q4 = np.round(ob['robot_0']['joint_state']['position'], 1)[6]
    q5 = np.round(ob['robot_0']['joint_state']['position'], 1)[7]
    q6 = np.round(ob['robot_0']['joint_state']['position'], 1)[8]
    q7 = np.round(ob['robot_0']['joint_state']['position'], 1)[9]
    flange = np.round(ob['robot_0']['joint_state']['position'], 1)[10]

    theta = [q1, q2, q3, q4, q5, q6, q7, flange]

    return theta


def get_endpoint_position(ob):
    # forward kinematics
    theta = get_joint_angles(ob)
    # DH parameters
    d = [1.333, 0, -0.316, 0, 0.384, 0, 0, 0.107]
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    alpha = [0, -np.pi / 2, np.pi/2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]

    # initialize transformation matrix to identity
    T = np.eye(4)

    # calculate cumulative transformation matrix
    for i in range(len(theta)):
        current_T = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]) * np.cos(alpha[i]), np.sin(theta[i]) * np.sin(alpha[i]), a[i] * np.cos(theta[i])],
            [np.sin(theta[i]), np.cos(theta[i]) * np.cos(alpha[i]), -np.cos(theta[i]) * np.sin(alpha[i]), a[i] * np.sin(theta[i])],
            [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
            [0, 0, 0, 1]
        ])
        T = np.dot(T, current_T)

    # end-effector position
    end_effector_pos = T[:3, 3]
    pose = [-end_effector_pos[0], -end_effector_pos[1], end_effector_pos[2]]
    return pose

# write a function: get_target_angles(target_position, ob) that changes the joint angles of only 1 and 2 and 6 to reach the target position using pybullet
def get_target_angles(target_position, ob):
    # forward kinematics
    theta = get_joint_angles(ob)
    # DH parameters
    d = [1.333, 0, -0.316, 0, 0.384, 0, 0, 0.107]
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    alpha = [0, -np.pi / 2, np.pi/2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]

    # initialize transformation matrix to identity
    T = np.eye(4)

    # calculate cumulative transformation matrix
    for i in range(len(theta)):
        current_T = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]) * np.cos(alpha[i]), np.sin(theta[i]) * np.sin(alpha[i]), a[i] * np.cos(theta[i])],
            [np.sin(theta[i]), np.cos(theta[i]) * np.cos(alpha[i]), -np.cos(theta[i]) * np.sin(alpha[i]), a[i] * np.sin(theta[i])],
            [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
            [0, 0, 0, 1]
        ])
        T = np.dot(T, current_T)

    # end-effector position
    end_effector_pos = T[:3, 3]
    pose = [-end_effector_pos[0], -end_effector_pos[1], end_effector_pos[2]]

    # inverse kinematics
    # calculate thetas
    target_theta = np.zeros(8)
    target_theta[0] = np.arctan2(pose[1], pose[0]) 
    target_theta[1] = np.arctan2(pose[2] - d[0], np.sqrt(pose[0]**2 + pose[1]**2)) - np.arctan2(a[3], d[2])
    target_theta[2] = 0
    target_theta[3] = np.arctan2(pose[2] - d[0], np.sqrt(pose[0]**2 + pose[1]**2)) - np.arctan2(a[3], d[2])
    target_theta[4] = 0
    target_theta[5] = 0
    target_theta[6] = 0
    target_theta[7] = 0


    return target_theta




# def get_target_angles(target_position, ob):
#     # forward kinematics
#     theta = get_joint_angles(ob)
#     # DH parameters
#     d = [1.333, 0, -0.316, 0, 0.384, 0, 0, 0.107]
#     a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
#     alpha = [0, -np.pi / 2, np.pi/2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]

#     # initialize transformation matrix to identity
#     T = np.eye(4)

#     # calculate cumulative transformation matrix
#     for i in range(len(theta)):
#         current_T = np.array([
#             [np.cos(theta[i]), -np.sin(theta[i]) * np.cos(alpha[i]), np.sin(theta[i]) * np.sin(alpha[i]), a[i] * np.cos(theta[i])],
#             [np.sin(theta[i]), np.cos(theta[i]) * np.cos(alpha[i]), -np.cos(theta[i]) * np.sin(alpha[i]), a[i] * np.sin(theta[i])],
#             [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
#             [0, 0, 0, 1]
#         ])
#         T = np.dot(T, current_T)

#     # end-effector position
#     end_effector_pos = T[:3, 3]
#     pose = [-end_effector_pos[0], -end_effector_pos[1], end_effector_pos[2]]

#     # inverse kinematics
#     # calculate thetas
#     target_theta = np.zeros(8)
#     target_theta[0] = np.arctan2(pose[1], pose[0]) 
    
#     target_theta[2] = 0
#     target_theta[3] = np.arctan2(pose[2] - d[0], np.sqrt(pose[0]**2 + pose[1]**2)) - np.arctan2(a[3], d[2])
#     target_theta[4] = 0
#     target_theta[5] = 0
#     target_theta[6] = 0
#     target_theta[7] = 0


#     return target_theta


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
    

def run_albert(n_steps=10000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            # hide the finger joints
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
    add_obstacles(env, [0.2, 0.2, 1.5], 0.01)

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
        pos=np.array([0, 0, -0.5*np.pi, 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0])
    )
    ob = ob[0]
    target_position = [0.2, 0.2, 1.5]
    target_angles = get_target_angles(target_position, ob)
    #loop door de steps heen, voer een actie uit met env.step(action)
    for _ in range(n_steps):

        # get the current joint angles
        
        
        

        # get the target joint angles
    

        # minimize the error using pid controller
        reached_1 = False
        reached_2 = False
        reached_3 = False

        while not reached_1:
            theta = get_joint_angles(ob)
            error_q1 = target_angles[0] - theta[0]
            pid_q1 = PIDcontroller(0.5, 0., 0.001, 0.01)
            action[2] = pid_q1.update(error_q1, 0.01)
            print(f'error_q1: {error_q1}') 
            ob, *_ = env.step(action)
            if np.abs(error_q1) < 0.02:
                reached_1 = True
                
        while not reached_2:
            theta = get_joint_angles(ob)
            error_q2 = np.abs(target_angles[1] - theta[1]) - np.pi
            # if the error is positive, the robot needs to move counterclockwise
            if target_angles[1] - theta[1] > 0:
                pid_q2 = PIDcontroller(0.5, 0., 0.001, 0.01)
                action[3] = pid_q2.update(error_q2, 0.01)
            elif target_angles[1] - theta[1] < 0:
                pid_q2 = PIDcontroller(0.5, 0., 0.001, 0.01)
                action[3] = -pid_q2.update(error_q2, 0.01)

            print(f'error_q2: {error_q2}')
            ob, *_ = env.step(action)
            if np.abs(error_q2) < 0.042:
                reached_2 = True

        while not reached_3:
            theta = get_joint_angles(ob)
            error_q4 = target_angles[3] - theta[3]
            pid_q4 = PIDcontroller(0.5, 0., 0.001, 0.01)
            action[5] = pid_q4.update(error_q4, 0.01)
            print(f'error_q4: {error_q4}')
            ob, *_ = env.step(action)
            if np.abs(error_q4) < 0.02:
                reached_3 = True
                print('reached')
                break


   
        
        


        # perform the action
        
    
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)