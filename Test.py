import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from global_path_planning_3d import RRTStar, Node
from sympy import symbols, atan2, sqrt, cos, sin

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
    add_obstacles(env, [0.10734279422187419, 0.4237077861241885, 1.04010572868276334], 0.05)

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
        pos=np.array([0, 0, -0.5*np.pi, 0.0, 0.0, 0.0, 0, 0.0, 0, 0])
    )
    ob = ob[0]
    #loop door de steps heen, voer een actie uit met env.step(action)
    for _ in range(n_steps):
        ob, *_ = env.step(action)

        pose = get_endpoint_position(ob)
        
        

        # print(f'x: {x}, y: {y}, z: {z}')

        # x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
        # y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
        # z = np.round(ob['robot_0']['joint_state']['position'], 1)[2]

        # print(f'x: {x}, y: {y}, z: {z}')
        # q_1_7 = inverse_kinematics(x, y, z)
        
        # action[0] = -0.3
        action[2] = 0.1 # joint 1
        ob, *_ = env.step(action)
    
        env.close()
        
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


# class ArmPIDcontroller():
#     def __init__(self, kp, ki, kd, dt):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.dt = dt
#         self.error_sum = 0
#         self.error_prev = 0

#     def update(self, error, dt):
#         self.error_sum += error * dt
#         error_diff = (error - self.error_prev) / dt
#         self.error_prev = error
#         return self.kp * error + self.ki * self.error_sum + self.kd * error_diff
    




def get_endpoint_position(ob):
    # forward kinematics
    theta = get_joint_angles(ob)
    # DH parameters
    d = [0.333, 0, -0.316, 0, 0.384, 0, 0, 0.107]
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    alpha = [0, -np.pi/2, np.pi, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]

    # transformation matrices
    T = np.zeros((4,4,8))
    for i in range(8):
        T[:,:,i] = np.array([
            [cos(theta[i]), -sin(theta[i]) * cos(alpha[i]), sin(theta[i]) * sin(alpha[i]), a[i] * cos(theta[i])],
            [sin(theta[i]), cos(theta[i]) * cos(alpha[i]), -cos(theta[i]) * sin(alpha[i]), a[i] * sin(theta[i])],
            [0, sin(alpha[i]), cos(alpha[i]), d[i]],
            [0, 0, 0, 1]
        ])
    
    # end-effector position
    end_effector_pos = np.dot(T[:,:,0], np.dot(T[:,:,1], np.dot(T[:,:,2], np.dot(T[:,:,3], np.dot(T[:,:,4], np.dot(T[:,:,5], T[:,:,6]))))))
    x, y, z = end_effector_pos[:3,3]
    pose = [-x, -y, z]
    return pose

# def inverse_kinematics(target_pose):




if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)