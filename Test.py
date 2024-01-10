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

    # a PID controller that controls the angle of the robot
    # Kp, Ki, Kd = PID parameters
    # dt = time step
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.error_sum = 0
        self.error_old = 0

    def update(self, error , dt):
        # update the PID controller
        # error = current error
        # dt = time step
        self.error_sum += error * dt
        error_diff = (error - self.error_old) / dt
        self.error_old = error
        return self.Kp * error + self.Ki * self.error_sum + self.Kd * error_diff

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
            spawn_rotation = 3.14,
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

    path_points = [[3, 0], [3, 3], [0, 3], [0,0]]

   

    #loop door de steps heen, voer een actie uit met env.step(action)
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        y = -np.round(ob['robot_0']['joint_state']['position'], 1)[0]
        x = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
        angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]

         # convert path points to local coordinates
        for i in range(len(path_points)):
            y = -np.round(ob['robot_0']['joint_state']['position'], 1)[0]
            x = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
            angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]

            # calibrate the local frame of the robot 0.04159265358979303 
            # (the robot is not perfectly aligned with the global frame)
            if angular > np.pi:
                angular -= 2*np.pi
            elif angular < -np.pi:
                angular += 2*np.pi

            x_local, y_local = transform_to_local_coordinates(x, y, angular, path_points[i][0], path_points[i][1])
            path_points[i] = [x_local, y_local]

        # calibrate the local frame of the robot 0.04159265358979303 
        # (the robot is not perfectly aligned with the global frame)
        if angular > np.pi:
            angular -= 2*np.pi
        elif angular < -np.pi:
            angular += 2*np.pi

        for target in path_points:
            reached = False
            print("next target: ", target)
            while not reached:
            # find the target angle by looking at the path points
                y = -np.round(ob['robot_0']['joint_state']['position'], 1)[0]
                x = np.round(ob['robot_0']['joint_state']['position'], 1)[1]
                angular = np.round(ob['robot_0']['joint_state']['position'], 1)[2]
                x_target = target[0]
                y_target = target[1]

                error = np.arctan2(y_target-y, x_target-x) - angular
                
            
                pid = PIDcontroller(0.5, 0.05, 0.1, 0.1)
                angular_velocity = pid.update(np.abs(error), 0.1)
                
                print("error: ", error)

                # rotate until target angle is reached
                action[0] = 0
                action[1] = angular_velocity
                ob, *_ = env.step(action)
                

                while error <= 0.042:
                    y = -np.round(ob['robot_0']['joint_state']['position'], 1)[0]
                    x = np.round(ob['robot_0']['joint_state']['position'], 1)[1]

                    action[1] = 0
                    action[0] = 0.5
                    ob, *_ = env.step(action)

                    # print robot position
                    print("x: ", x)
                    print("y: ", y)


                    # check if the robot is close enough to the target point
                    print("distance to target: ", np.sqrt((x-x_target)**2 + (y-y_target)**2))
                    if np.sqrt((x-x_target)**2 + (y-y_target)**2) <= 0.3:
                        path_points.pop(0)
                        reached = True
                        print("target reached")
                        break
                    



        
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
