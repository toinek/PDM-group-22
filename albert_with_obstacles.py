import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from mpscenes.obstacles.urdf_obstacle import UrdfObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle


def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    
    boundary = 16

    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_offset = np.array([boundary/2 - 1.5, boundary/2 - 1.5, 0.15]),
            spawn_rotation = 0,
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )

    # create dictionary of 4 walls that will spawn around the albert robot
    wall_length = boundary
    wall_obstacles_dicts = [
        {
            'type': 'box', 
            'geometry': {
                'position': [wall_length/2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': 0.1
            },
            'high': {
                'position' : [wall_length/2.0, 0.0, 0.4],
                'width': wall_length,
                'height': 0.8,
                'length': 0.1,
            },
            'low': {
                'position' : [wall_length/2.0, 0.0, 0.4],
                'width': wall_length,
                'height': 0.8,
                'length': 0.1,
            },
        },
        {
            'type': 'box', 
            'geometry': {
                'position': [0.0, wall_length/2.0, 0.4], 'width': 0.1, 'height': 0.8, 'length': wall_length
            },
            'high': {
                'position' : [0.0, wall_length/2.0, 0.4],
                'width': 0.1,
                'height': 0.8,
                'length': wall_length,
            },
            'low': {
                'position' : [0.0, wall_length/2.0, 0.4],
                'width': 0.1,
                'height': 0.8,
                'length': wall_length,
            },
        },
        {
            'type': 'box', 
            'geometry': {
                'position': [0.0, -wall_length/2.0, 0.4], 'width': 0.1, 'height': 0.8, 'length': wall_length
            },
            'high': {
                'position' : [0.0, -wall_length/2.0, 0.4],
                'width': 0.1,
                'height': 0.8,
                'length': wall_length,
            },
            'low': {
                'position' : [0.0, -wall_length/2.0, 0.4],
                'width': 0.1,
                'height': 0.8,
                'length': wall_length,
            },
        },
        {
            'type': 'box', 
            'geometry': {
                'position': [-wall_length/2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': 0.1
            },
            'high': {
                'position' : [-wall_length/2.0, 0.0, 0.4],
                'width': wall_length,
                'height': 0.8,
                'length': 0.1,
            },
            'low': {
                'position' : [-wall_length/2.0, 0.0, 0.4],
                'width': wall_length,
                'height': 0.8,
                'length': 0.1,
            },
        },
    ]


    # add walls to the environment
    wall_obstacles = [BoxObstacle(name=f"wall_{i}", content_dict=obst_dict) for i, obst_dict in enumerate(wall_obstacles_dicts)]
    for i in range(len(wall_obstacles)):
        env.add_obstacle(wall_obstacles[i])

    # defining an action
    action = np.zeros(env.n())
    action[0] = 0.2
    action[1] = 0
    action[2] = 0
    action[3] = 0
    action[4] = 0
    action[5] = 0
    action[6] = 0
    action[7] = 0
    action[8] = 0
    action[9] = 0
    action[10] = 0
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )

    for _ in range(n_steps):
        for _ in range(n_steps):
            ob, *_ = env.step(action)
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)