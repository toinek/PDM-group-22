import warnings
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
    theta = (np.round(ob['robot_0']['joint_state']['position'], 1)[4] + 1.189218407)
    return x, y, angular, theta

def run_albert(n_steps=100000, render=False, goal=True, obstacles=True):

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
    obstacle_adder.add_spheres()
    obstacle_adder.add_walls(boundary)

    # Algorithm inputs
    bounds = {'xmin': -0, 'xmax': boundary/2, 'ymin': -0, 'ymax': boundary/2, 'zmin': 0, 'zmax': 0}
    start_base = [0,0,0]
    goal_base = [7,5,0]
    obstacles = {i:(env.get_obstacles()[obstacle])._content_dict['geometry'] for i, obstacle in enumerate(env.get_obstacles()) if (env.get_obstacles()[obstacle])._content_dict['type'] == 'sphere'}

    # Initialize and run algorithm
    rrt_star_base = RRTStar(start_base, goal_base, bounds, obstacles, max_iter=100000)
    path_points = rrt_star_base.full_run()[::-1]
    illustrate_algorithm_3d(rrt_star_base)

    # Extract positionts of the nodes
    path_points = [rrt_star_base.nodes[node].position for node in path_points if node != 0]
    print(f'path_points: {path_points}')

    # Visualize as test
    for point in path_points:
        rounded_points = np.round(point, 2).astype(float)
        print(f'rounded_points: {rounded_points}')
        if rounded_points[1] != 0:
            x = float(rounded_points[0])
            y = float(rounded_points[1])
            z = float(rounded_points[2])
            add_obstacles(env, [x,y,z+2], 0.1)

    # Initialize action
    action = np.zeros(env.n())

    # Perform n steps and follow the computed path
    link_length = 0.649864421

    base_control = PIDControllerBase(path_points, 0.05, 0., 0.001, 0.01)

    arm_pid_controller = PIDControllerArm(1, 0., 0.001, 0.01)
    arm_kinematics = RobotArm(link_length, [-0.120625, 0, 0.958], arm_pid_controller)
    previous_error = np.inf
    for _ in range(n_steps):
        # Obtain robot configuration
        robot_config = get_robot_config(ob)
        x_robot, y_robot, angular, theta = robot_config

        # Control the base
        forward_velocity, angular_velocity = base_control.follow_path(robot_config)
        action[0] = forward_velocity * 1.5
        action[1] = angular_velocity * 10
        #print(f'forward velocity: {forward_velocity}, angular velocity: {angular_velocity}')

        # If the base reached the goal, control the arm
        goal_within_range = arm_kinematics.goal_within_reach(x_robot, y_robot, angular, path_points[0])
        print(path_points)
        if goal_within_range:
            action[0] = 0
            action[1] = 0
            base_control.target_reached = True
        if not arm_kinematics.target_reached and base_control.target_reached and len(path_points) == 1:
            goal = path_points[0]
            theta = (np.round(ob['robot_0']['joint_state']['position'], 1)[
                         4] + 1.189218407)
            q, previous_error = arm_kinematics.follow_arm_path(x_robot, y_robot, angular, theta, goal, previous_error)
            print(f'q: {q}')
            action[3] = q
        if arm_kinematics.target_reached:
            action[3] = 0
            arm_kinematics.target_reached = False
            base_control.target_reached = False
            path_points.pop(0)
        ob, *_ = env.step(action)

    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)