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
    # print the obstacles within the environment
    for obstacle in env.get_obstacles():
        print(f"Obstacles in the environment: {env.get_obstacles()[obstacle].__dict__}")
        print(f"Obstacle position: {(env.get_obstacles()[obstacle])._content_dict['geometry']}")

    bounds = {'xmin': 0, 'xmax': 7, 'ymin': 0, 'ymax': 7, 'zmin': 0, 'zmax': 2}
    start = [0,0,0]
    goal = [5,5,0]
    obstacles = {i:(env.get_obstacles()[obstacle])._content_dict['geometry'] for i, obstacle in enumerate(env.get_obstacles())}
    print(obstacles)
    rrt_star_base = RRTStar(start, goal, bounds, obstacles, max_iter=100000)
    path_points = rrt_star_base.full_run()

    path_points = [rrt_star_base.nodes[node].position for node in path_points]
    path_points.append([3,2,0])
    for point in path_points:
        rounded_points = np.round(point, 2).astype(float)
        print(f'rounded_points: {rounded_points}')
        if rounded_points[1] != 0:
            x = float(rounded_points[0])
            y = float(rounded_points[1])
            z = float(rounded_points[2])
            add_obstacles(env, [x,y,z+2], 0.1)
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
        # ob, *_ = env.step(action)
        x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
        y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
        angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]

        # Use the pid controller to control the angular velocity to aim at the next point and drive towards it, not looping over every target all the time
        for target in path_points:
            if target[0] != 0:
                angle_reached = False
                print("next target: ", target)
                while not angle_reached:
                    robot_config = get_robot_config(ob)
                    x = robot_config[0]
                    y = robot_config[1]
                    robot_angle = robot_config[2]
                    pid = PIDcontroller(1, 0.0, 0.001, 0.01)
                    desired_angle = np.arctan2(target[1] - y, target[0] - x)
                    angle_error = desired_angle - robot_angle
                    angular_vel = pid.update(angle_error, 0.01)
                    action[0] = 0
                    action[1] = angular_vel
                    ob, *_ = env.step(action)
                    if abs(angle_error) < 0.05:
                        angle_reached = True
                        print("angle reached")
                        target_reached = False
                        while not target_reached:
                            robot_config = get_robot_config(ob)
                            x = robot_config[0]
                            y = robot_config[1]
                            print(x,y)
                            target_error = np.sqrt((target[0] - x)**2 + (target[1] - y)**2)
                            action[0] = 0.5
                            action[1] = 0
                            ob, *_ = env.step(action)
                            if target_error < 0.5:
                                print('target reached')
                                target_reached = True
                                break




                # while error < 0.02:
                #     x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
                #     y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
                #     print(np.round(ob['robot_0']['joint_state']['position'], 1))
                #     angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]
                #     # print("test")
                #     action[1] = 0
                #     action[0] = 0.5
                #     ob, *_ = env.step(action)
                #
                #     # move to next target when previous target is reached within 0.1m, taking into account that coordinates can be negative
                #     if abs(target[0]) - abs(x) < 0.1 and abs(target[1]) - abs(y) < 0.2:
                #         action[0] = 0
                #         ob, *_ = env.step(action)
                #         print('target reached')
                #         reached = True
                #         break

                

                


            
                
            


        # print(f'Albert position: {x}, {y}, angle: {angular}')
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)