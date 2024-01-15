import warnings
import argparse
import time
import gymnasium as gym
import numpy as np

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from global_path_planning_3d import RRTStar, illustrate_algorithm_3d
from local_path_planning import PIDControllerBase
from local_arm_control import PIDControllerArm
from add_obstacles import ObstacleAdder
from kinematics import RobotArm

def add_obstacles(env, pos, radius, rgba):
    sphere_obst_dict = {
        "type": "sphere",
        'movable': False,
        "geometry": {"position": pos, "radius": radius},
        "rgba": rgba,
    }
    sphere_obst = SphereObstacle(name=f'obstacle_{pos[0]}_{pos[1]}_{pos[2]}', content_dict=sphere_obst_dict)
    env.add_obstacle(sphere_obst)

def plot_path(env, path):
    for point in path:
        rounded_points = np.round(point, 2).astype(float)
        if rounded_points[1] != 0:
            x = float(rounded_points[0])
            y = float(rounded_points[1])
            z = float(rounded_points[2])
            add_obstacles(env, [x,y,2], 0.1, [0.0, 1.0, 0.0, 1.0])

def get_robot_config(ob):
    x = np.round(ob['robot_0']['joint_state']['position'], 1)[0]
    y = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
    angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]
    theta = (np.round(ob['robot_0']['joint_state']['position'], 1)[4] + 1.189218407)
    return x, y, angular, theta

def run_albert(rrt_star, start_base, hard, n_steps=100000, render=True):

    # Create robot and environment
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

    # initial position
    ob = env.reset(
        pos=np.array([0, 0, -0.5*np.pi, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    ob = ob[0]

    # Add obstacles to the environment
    boundary = 16
    obstacle_adder = ObstacleAdder(env)
    obstacle_adder.add_spheres(hard)
    obstacle_adder.add_walls(boundary)

    # Algorithm inputs
    if hard:
        goal_base = [7, 7, 0]
    else:
        goal_base = [5, 5, 0]
    bounds = {'xmin': 0, 'xmax': 8, 'ymin': 0, 'ymax': 8, 'zmin': 0, 'zmax': 2}
    obstacles = {i:(env.get_obstacles()[obstacle])._content_dict['geometry'] for i, obstacle in enumerate(env.get_obstacles()) if (env.get_obstacles()[obstacle])._content_dict['type'] == 'sphere'}

    # Initialize and run algorithm
    rrt_star_base = RRTStar(start_base, goal_base, bounds, obstacles, epsilon=0.4, max_iter=100000, two_dim=True, star=rrt_star)
    start_time = time.time()
    path_points, path_cost = rrt_star_base.full_run()
    end_time = time.time()
    execution_time = end_time-start_time
    print(f'execution time: {execution_time}')
    path_points = path_points[::-1]
    illustrate_algorithm_3d(rrt_star_base)

    # Initialize action
    action = np.zeros(env.n())

    # Initialize the base control
    base_control = PIDControllerBase(path_points, 1, 0., 0.01, 0.01)
    base_control.interpolate_path(path_points, rrt_star_base.nodes)
    plot_path(env, base_control.path)

    # Initialize the arm control
    link_length = 0.649864421
    previous_error = np.inf
    arm_goal = [goal_base[0],goal_base[1],0.8]#rrt_star_base.nodes[path_points[-1]].position
    add_obstacles(env, arm_goal, 0.1, [1.0, 1.0, 0.0, 1.0])
    print(f'arm_goal: {arm_goal}')
    arm_pid_controller = PIDControllerArm(1, 0., 0.001, 0.01)
    arm_kinematics = RobotArm(link_length, [-0.120625, 0, 0.958], arm_pid_controller)

    # Run the simulation for n steps
    for _ in range(n_steps):
        # Obtain robot configuration
        robot_config = get_robot_config(ob)
        x_robot, y_robot, angular, theta = robot_config

        # Control the base
        forward_velocity, angular_velocity = base_control.follow_path(robot_config)
        action[0] = np.clip(forward_velocity, -0.55, 0.55)
        action[1] = np.clip(angular_velocity, a_min=-0.5, a_max=0.5)

        # If the base reached the goal, control the arm
        goal_within_range = arm_kinematics.goal_within_reach(x_robot, y_robot, angular, arm_goal)
        if goal_within_range:
            action[0] = 0
            action[1] = 0
            base_control.target_reached = True
        if not arm_kinematics.target_reached and base_control.target_reached and len(base_control.path) == 0:
            q, previous_error = arm_kinematics.follow_arm_path(x_robot, y_robot, angular, theta, arm_goal, previous_error)
            action[3] = q
        if arm_kinematics.target_reached:
            action[3] = 0
            arm_kinematics.target_reached = False
            base_control.target_reached = False
        ob, *_ = env.step(action)

    env.close()


def run_from_command_line(args):
    #start_base = [float(val) for val in args.start_base.split(',')]
    render = args.render
    hard = args.hard
    rrt_star = args.rrt_star
    start_base = [0, 0, 0]
    with warnings.catch_warnings():
        run_albert(start_base=start_base, render=render, hard=hard, rrt_star=rrt_star)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robotic Motion Planning and Decision Making')

    # Add command-line arguments
    parser.add_argument('--render', action='store_true', help='Enable visualization')
    parser.add_argument('--hard', action='store_true', help='Set to True for a harder goal position')
    parser.add_argument('--rrt_star', action='store_true', help='Use RRT* algorithm for global path planning')

    args = parser.parse_args()

    run_from_command_line(args)




# if __name__ == "__main__":
#     show_warnings = False
#     warning_flag = "default" if show_warnings else "ignore"
#     with warnings.catch_warnings():
#         start_base = [0, 0, 0]
#         warnings.filterwarnings(warning_flag)
#         run_albert(start_base=start_base, render=True, hard=True, rrt_star=True)


