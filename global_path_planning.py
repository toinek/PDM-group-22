import numpy as np

def euclidean_distance(q1, q2):
    # Compute the Euclidean distance between two points
    # Return the distance
    distance = np.linalg.norm(np.array(q1) - np.array(q2))
    return distance


class RRT:
    def __init__(self, start, goal, bounds, obstacles, step_size=0.1, max_iter=1000):
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [self.start]

    def check_sample_in_obstacle(self, q_rand, epsilon):
        for obstacle in self.obstacles:
            obstacle_position_2d = self.obstacles[obstacle]['position'][0], self.obstacles[obstacle]['position'][1]
            distance = euclidean_distance(q_rand, obstacle_position_2d)
            if distance < self.obstacles[obstacle]['radius'] + epsilon:
                print("Collision detected")
                return True
        return False

    def sample_random_config(self):
        # Sample a random configuration q_rand in the state space
        # You can sample from a uniform distribution within the state bounds
        # Return the sampled configuration
        in_collision = True
        while in_collision:
            q_rand = np.random.uniform([self.bounds['xmin'], self.bounds['ymin']],
                                               [self.bounds['xmax'], self.bounds['ymax']])
            print(q_rand)
            in_collision = self.check_sample_in_obstacle(q_rand, 0.5)

        return q_rand

    def nearest_neighbour(self, q_rand):
        # Return the nearest neighbor of q_rand from the tree
        min_distance = float('inf')
        nearest_neighbour = None
        for node in self.tree:
            distance = euclidean_distance(node, q_rand)
            if distance < min_distance:
                min_distance = distance
                nearest_neighbour = node
        return nearest_neighbour

    def check_collision_in_path(self, q1, q2):
        # Check for collisions between two configurations q1 and q2
        for obstacle in self.obstacles:
            cx, cy, r = self.obstacles[obstacle]['position'][0], self.obstacles[obstacle]['position'][1], self.obstacles[obstacle]['radius']
            x1, y1, x2, y2 = q1[0], q1[1], q2[0], q2[1]

            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            A = 1 + m**2
            B = 2 * (m * c - m * cy - cx)
            C = cx**2 + cy**2 + c**2 - 2 * cy * c - r**2

            discriminant = B**2 - 4 * A * C
            print(discriminant)
            x_intersects = (-B + np.sqrt(discriminant)) / (2 * A)
            y_intersects = m * x_intersects + c
            print(self.obstacles[obstacle])
            print(x_intersects, y_intersects)

        return False

    def extend(self, q_near, q_rand):
        # Extend the tree from configuration q_near towards q_rand
        # Return the new configuration that is reached by this extension
        # Placeholder implementation
        return q_near

    def check_goal_reached(self, q):
        # Check if the goal has been reached
        # Return True if goal has been reached, False otherwise
        # Placeholder implementation
        return False

obstacles = {1:{'position': [1, 0, 0], 'radius': 0.5}}
test_rrt = RRT(start=[0, 0, 0], goal=[1, 1, 1], bounds={'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'zmin': 0, 'zmax': 0}, obstacles=obstacles)
test_rrt.sample_random_config()
test_rrt.check_collision_in_path([0, 0], [2, 0])

for obstacle in obstacles:
    print(obstacle)