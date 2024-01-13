import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from global_path_planning_3d import RRTStar
from local_path_planning import PIDControllerBase
from local_arm_control import PIDControllerArm
from add_obstacles import ObstacleAdder

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
    # bounds = {'xmin': -boundary/2, 'xmax': boundary/2, 'ymin': -boundary/2, 'ymax': boundary/2, 'zmin': 0, 'zmax': 3}
    # start_base = [0,0,0]
    # goal_base = [6,5,0]
    # obstacles = {i:(env.get_obstacles()[obstacle])._content_dict['geometry'] for i, obstacle in enumerate(env.get_obstacles()) if (env.get_obstacles()[obstacle])._content_dict['type'] == 'sphere'}
    #
    # # Initialize and run algorithm
    # rrt_star_base = RRTStar(start_base, goal_base, bounds, obstacles, max_iter=100000)
    # path_points = rrt_star_base.full_run()
    #
    # # Extract positionts of the nodes
    # path_points = [rrt_star_base.nodes[node].position for node in path_points]
    #
    # # Visualize as test
    # path_points.append([3,2,0])
    # for point in path_points:
    #     rounded_points = np.round(point, 2).astype(float)
    #     print(f'rounded_points: {rounded_points}')
    #     if rounded_points[1] != 0:
    #         x = float(rounded_points[0])
    #         y = float(rounded_points[1])
    #         z = float(rounded_points[2])
    #         add_obstacles(env, [x,y,z+2], 0.1)

    # Initialize action
    action = np.zeros(env.n())

    # Perform n steps and follow the computed path
    # base_control = PIDControllerBase(path_points, 0.05, 0., 0.001, 0.01)

    arm_pid_controller = PIDControllerArm(0.5, 0., 0.001, 0.01)
    #add_obstacles(env, goal, 0.01)
    for _ in range(n_steps):
        # robot_config = get_robot_config(ob)
        # forward_velocity, angular_velocity = base_control.follow_path(robot_config)
        # action[0] = forward_velocity
        # action[1] = angular_velocity
        # print(f'forward velocity: {forward_velocity}, angular velocity: {angular_velocity}')

        ob, *_ = env.step(action)

        # Control the arm
        end_pos = [0.4825, 0, 1.2]
        
        # add_obstacles(env, end_pos, 0.01)
        neutral_joint_pos = [-0.120625, 0, 0.958]
        link_length = 0.649864421
        goal = [-0.120625, 0, 0.958 + link_length-0.2]
        
        x_robot,y_robot,angular = get_robot_config(ob)[0], get_robot_config(ob)[1], get_robot_config(ob)[2]

        # x_joint = x_robot - (link_length - (np.cos(angular)*link_length))
        # y_joint = y_robot + np.sin(angular) * link_length
        x_joint = -(link_length - (np.cos(angular)*link_length))
        y_joint = np.sin(angular) * link_length

        # add_obstacles(env, [float(x_joint), float(y_joint), 1.8], 0.01)
        joint_pos = [x_joint, y_joint, neutral_joint_pos[2]]

        theta = (np.round(ob['robot_0']['joint_state']['position'], 1)[4] + 0.9)#1.189218407)
        end_effector_pos = [(x_joint + link_length*np.sin(theta)), y_joint, (neutral_joint_pos[2] + link_length*np.cos(theta)- 0.2)]

        # Calculate the error between the endpoint and goal
        error = np.sqrt((end_effector_pos[0] - goal[0])**2 + (end_effector_pos[1] - goal[1])**2 + (end_effector_pos[2] - goal[2])**2)
        print(f'error: {error}')

        angular_vel = arm_pid_controller.get_angular_vel(error)
        # angular velocity should be negative if the end effector is to the left of the goal
        if end_effector_pos[0] > goal[0]:
            angular_vel = -angular_vel


        action[3] = angular_vel


        # print(f'error_x: {error_x}, error_y: {error_y}, error_z: {error_z}')
        # print(f'error: {error}')




        # action[3] = angular_vel
            # x = float(np.round(end_effector_pos[0], 2))
            # y = float(np.round(end_effector_pos[1], 2))
            # z = float(np.round(end_effector_pos[2], 2))
            # add_obstacles(env, [x,y,z], 0.01)
    
        ob, *_ = env.step(action)

    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)