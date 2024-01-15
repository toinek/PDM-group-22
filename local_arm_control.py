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

    def get_angular_vel(self, error):
        self.error_sum += error * self.dt
        error_diff = (error - self.error_prev) / self.dt
        self.error_prev = error
        angular_vel = self.kp * error + self.ki * self.error_sum + self.kd * error_diff
        return angular_vel
    
    

