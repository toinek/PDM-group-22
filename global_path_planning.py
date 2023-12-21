import gymnasium as gym
from urdfenvs.robots.generic_urdf import GenericUrdfReacher

robots = [GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel")]
env = gym.make('urdf-env-v0', dt=0.01, robots=robots, render=True)
env.reconfigure_camera(2.0, 0.0, -90.01, (0, 0, 0))
env.reset()

while True:
    for _ in range(1000):
        env.render()