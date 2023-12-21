import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


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
    action = np.zeros(env.n())
    action[0] = 0.2
    action[1] = 0.0
    action[5] = -0.1
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
import pybullet as p
import numpy as np
import random

class RRT:
    def __init__(self, start, goal, step_size, max_iter):
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [start]

    def is_goal_reached(self, node):
        # Define your goal condition here
        return np.linalg.norm(np.array(node) - np.array(self.goal)) < self.step_size

    def extend(self, q_near, q_rand):
        # Extend the tree from q_near towards q_rand
        direction = np.array(q_rand) - np.array(q_near)
        magnitude = np.linalg.norm(direction)
        unit_vector = direction / magnitude if magnitude > 0 else direction
        q_new = [q_near[i] + self.step_size * unit_vector[i] for i in range(len(q_near))]
        return q_new

    def nearest_neighbor(self, q_rand):
        # Find the nearest node in the tree to q_rand
        distances = [np.linalg.norm(np.array(q_rand) - np.array(node)) for node in self.tree]
        min_index = np.argmin(distances)
        return self.tree[min_index]

    def generate_random_node(self):
        # Generate a random configuration in the workspace
        # Adjust this based on the workspace and joint limits of the Albert robot
        return [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)]

    def rrt_algorithm(self):
        for _ in range(self.max_iter):
            q_rand = self.generate_random_node()
            q_near = self.nearest_neighbor(q_rand)
            q_new = self.extend(q_near, q_rand)

            # Check for collisions and validity of q_new
            if not self.is_collision(q_near, q_new):
                self.tree.append(q_new)

                if self.is_goal_reached(q_new):
                    return self.construct_path()

        return None  # Return None if no path is found within the maximum iterations

    def is_collision(self, q1, q2):
        # Check for collisions between two configurations q1 and q2
        # You'll need to use the PyBullet physics engine functions here
        # Implement collision checking based on the specifics of your environment
        # Example: Use p.rayTest or p.getClosestPoints functions to check for collisions

        # Placeholder implementation
        return False

    def construct_path(self):
        # Reconstruct the path from the goal to the start using the tree
        path = [self.goal]
        current = self.goal
        while current != self.start:
            nearest = self.nearest_neighbor(current)
            path.append(nearest)
            current = nearest

        path.reverse()
        return path

# Example usage
if __name__ == "__main__":
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(1)

    # Load your URDF file for the Albert robot
    robot_urdf_path = "path/to/your/albert/robot.urdf"
    robot_id = p.loadURDF(robot_urdf_path, [0, 0, 0])

    # Set the start and goal configurations
    start_config = [0, 0, 0]
    goal_config = [1, 1, 1]

    # Initialize the RRT planner
    rrt_planner = RRT(start_config, goal_config, step_size=0.1, max_iter=1000)

    # Run the RRT algorithm
    path = rrt_planner.rrt_algorithm()

    # Print the path
    if path:
        print("Path found:", path)
    else:
        print("No path found within the maximum iterations.")

    # Close the simulation
    p.disconnect()
