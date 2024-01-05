import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle


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

    # Adding a sphere obstacle
    for i in range(10):
        pos = [2+0.*i, 0.1*i+2, 1]
        radius = 0.5
        sphere_obst_dict = {
            "type": "sphere",
            'movable': False,
            "geometry": {"position": pos, "radius": radius},
        }
        sphere_obst = SphereObstacle(name=f'obstacle_{pos[0]}_{pos[1]}_{pos[2]}', content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)

    # obstacles = env.get_obstacles()
    # for i in obstacles:
    #     print(i)
    # print(obstacles[i].__dict__)
    # print(obstacles[i]._config['geometry'])
    # defining an action
    action = np.zeros(env.n())
    action[0] = 0.2
    action[1] = 1
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
        ob, *_ = env.step(action)
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)