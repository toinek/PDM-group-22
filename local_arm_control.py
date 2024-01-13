import numpy as np
class PIDControllerArm():

    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.error_sum = 0
        self.error_prev = 0
        self.angle_reached = False
        self.target_reached = False

