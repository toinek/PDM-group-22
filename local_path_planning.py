import numpy as np
class PIDControllerBase:

    # a pid controller to control the angular velocity of the robot, aiming at the next target
    def __init__(self, path, kp, ki, kd, dt):
        self.path = path
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.error_sum = 0
        self.error_prev = 0
        self.angle_reached = False
        self.target_reached = False

    def update(self, error):
        self.error_sum += error * self.dt
        error_diff = (error - self.error_prev) / self.dt
        if error != self.error_prev:
            self.error_prev = error
        return self.kp * error + self.ki * self.error_sum + self.kd * error_diff

    def control_angle(self, robot_ob, target):
        x, y, robot_angle = robot_ob[0], robot_ob[1], robot_ob[2]
        desired_angle = np.arctan2(target[1] - y, target[0] - x)
        angle_error = desired_angle - robot_angle
        angular_vel = self.update(angle_error) #0.4
        if abs(angle_error) < 0.1:
            print("base angle reached")
            self.angle_reached = True
        return angular_vel

    def control_velocity(self, robot_ob, target):
        x, y, robot_angle = robot_ob[0], robot_ob[1], robot_ob[2]
        target_error = np.sqrt((target[0] - x) ** 2 + (target[1] - y) ** 2)
        forward_velocity = self.update(target_error) #0.5
        if target_error < 0.5:
            print('base target reached')
            self.target_reached = True
        return forward_velocity

    def interpolate_path(self, path, nodes):
        interpolated_points = []
        for i in range(len(path) - 1):
            start_node = nodes[path[i]].position
            end_node = nodes[path[i + 1]].position

            # Interpolate between start_node and end_node
            for t in np.linspace(0, 1, 5):
                interpolated_point = (1 - t) * np.array(start_node) + t * np.array(end_node)
                interpolated_points.append(interpolated_point)
        self.path = interpolated_points

    def follow_path(self, robot_ob):
        if len(self.path) == 0:
            return 0, 0
        target = self.path[0]
        if not self.angle_reached:
            angular_vel = self.control_angle(robot_ob, target)
            forward_velocity = 0
            return forward_velocity, angular_vel
        if not self.target_reached:
            angular_vel = 0
            forward_velocity = self.control_velocity(robot_ob, target)
        else:
            forward_velocity = 0
            angular_vel = 0
            self.angle_reached = False
            self.target_reached = False
            self.path.pop(0)
            print('target reached')
        return forward_velocity, angular_vel


