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
    add_obstacles(env, [3.0, 3.0, 0], 0.1)

    # Rotate the robot's frame by +90 degrees around the z-axis
    initial_position = np.array([0, 0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    initial_position[2] += np.pi / 2  # Add +90 degrees rotation

    # print the obstacles within the environment
    for obstacle in env.get_obstacles():
        print(f"Obstacles in the environment: {env.get_obstacles()[obstacle].__dict__}")
        print(f"Obstacle position: {(env.get_obstacles()[obstacle])._content_dict['geometry']}")

    action = np.zeros(env.n())
    action[0] = 0 # forward velocity
    action[1] = 0 # angular velocity
    # ... (rest of the action setup)

    # initial position
    ob = env.reset(pos=initial_position)

    # loop through the steps, execute an action with env.step(action)
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
        y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
        angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]

# print(f'Albert position: {x}, {y}, angle: {angular}')
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
