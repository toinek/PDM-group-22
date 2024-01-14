import numpy as np

class RobotArm:

    def __init__(self, link_length, joint_offset, pid_controller):
        self.link_length = link_length
        self.joint_offset = joint_offset
        self.pid_controller = pid_controller
        self.target_reached = False
    def get_joint_pos(self, x_robot, y_robot, angular):
        x_joint = x_robot - np.cos(angular) * -self.joint_offset[0]
        y_joint = y_robot - np.sin(angular) * -self.joint_offset[0]
        z_joint = self.joint_offset[2]
        return x_joint, y_joint, z_joint

    def forward_kinematics(self, x_robot, y_robot, angular, theta):
        x_joint, y_joint, z_joint = self.get_joint_pos(x_robot, y_robot, angular)
        arm_length = np.cos(theta) * self.link_length
        end_effector_pos = [(x_joint + arm_length * np.cos(angular)),
                            (y_joint + arm_length * np.sin(angular)),
                            (self.joint_offset[2] + np.sin(theta) * arm_length)]
        return end_effector_pos

    def goal_within_reach(self, x_robot, y_robot, angular, goal):
        x_joint, y_joint, z_joint = self.get_joint_pos(x_robot, y_robot, angular)
        goal_within_range = np.sqrt((x_joint - goal[0]) ** 2 + (y_joint - goal[1]) ** 2 + (
                z_joint - goal[2]) ** 2) <= self.link_length
        return goal_within_range

    def follow_arm_path(self, x_robot, y_robot, angular, theta, goal, previous_error):
        self.target_reached = False
        end_effector_pos = self.forward_kinematics(x_robot, y_robot, angular, theta)
        # error = np.sqrt((goal[0] - end_effector_pos[0])**2 + (goal[1] - end_effector_pos[1])**2 + (goal[2] - end_effector_pos[2])**2)
        if end_effector_pos[2] < goal[2]:
            error = goal[2] - end_effector_pos[2]
            print(f'error: {error}')
            q = -self.pid_controller.get_angular_vel(error)
        else:
            error = end_effector_pos[2] - goal[2]
            print(f'error: {error}')
            q = self.pid_controller.get_angular_vel(error)
        if error > previous_error:
            self.target_reached = True
            q = 0
            print('reached arm')
        if error < 0.05:
            self.target_reached = True
            q = 0
            print('reached arm')
        print(f'previous error: {previous_error}')
        previous_error = error
        return q, previous_error

