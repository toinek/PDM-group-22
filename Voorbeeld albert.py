import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

def add_obstacles(env, pos, radius):
    sphere_obst_dict = {
        "type": "sphere",
        'movable': False,
        "geometry": {"position": pos, "radius": radius},
    }
    sphere_obst = SphereObstacle(name=f'obstacle_{pos[0]}_{pos[1]}_{pos[2]}', content_dict=sphere_obst_dict)
    env.add_obstacle(sphere_obst)

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

def transform_to_local_coordinates(x, y, angle, target_x, target_y):
    # transform the global coordinates to local coordinates
    # x, y, angle = robot position and angle
    # target_x, target_y = target position
    # returns the local coordinates of the target
    x_local = (target_x - x) * np.cos(angle) + (target_y - y) * np.sin(angle)
    y_local = -(target_x - x) * np.sin(angle) + (target_y - y) * np.cos(angle)
    return x_local, y_local

def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 0,
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )

    add_obstacles(env, [1.5, 0, 0], 0.5)

    # print the obstacles within the environment
    for obstacle in env.get_obstacles():
        print(f"Obstacles in the environment: {env.get_obstacles()[obstacle].__dict__}")
        print(f"Obstacle position: {(env.get_obstacles()[obstacle])._content_dict['geometry']}")

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
    action[9] = 1 # gripper -> snap deze nog niet helemaal
    action[10] = 0 # gripper -> snap deze nog niet helemaal

    # initial position
    ob = env.reset(
        pos=np.array([0, 0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )

    #loop door de steps heen, voer een actie uit met env.step(action)
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
        y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
        angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]

        path_points = [(3.0, 0.0), (3.0, -3.0), (0.0, 3.0), (0.0, 0.0)]

        # Use the pid controller to control the angular velocity to aim at the next point and drive towards it, not looping over every target all the time
        for target in path_points:
            pid = PIDcontroller(0.5, 0.0, 0.001, 0.01)
            reached = False
            print("next target: ", target)

            while not reached:
                x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
                y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
                angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]
        
                # transform the local coordinates to global coordinates where the z axis should be rotated 90 degrees
                x_local, y_local = transform_to_local_coordinates(x, y, angular, target[1], target[0])
                print("local coordinates: ", x_local, y_local)
                error = np.arctan2(y_local, x_local) + angular

                print("error: ", error)

                angular_vel = pid.update(error, 0.1)
                action[1] = angular_vel
                ob, *_ = env.step(action)

                while 0 < np.abs(error) < 0.02:
                    x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
                    y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
                    angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]
                    # print("test")
                    action[1] = 0
                    action[0] = 0.8
                    ob, *_ = env.step(action)
                    print(np.sqrt((x - target[1])**2 + (y - target[0])**2))
                    if np.sqrt((x - target[1])**2 + (y - target[0])**2) <= 0.3:
                        action[0] = 0
                        ob, *_ = env.step(action)
                        print('----------------------------------------------\n target reached\n----------------------------------------------')
                        reached = True
                        break
                    break
                break

                

                


            
                
            


        # print(f'Albert position: {x}, {y}, angle: {angular}')
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
