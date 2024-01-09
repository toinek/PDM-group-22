import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def euclidean_distance(q1, q2):
    # Compute the Euclidean distance between two points
    # Return the distance
    distance = np.linalg.norm(np.array(q1) - np.array(q2))
    return distance

class Node:
    def __init__(self, number, position, parent):
        self.number = number
        self.position = position
        self.parent = parent

    def __repr__(self):
        return f'Node {self.number} : ({self.position}, {self.parent})'


class RRT:
    def __init__(self, start, goal, bounds, obstacles, step_size=0.1, max_iter=1000):
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = {0: Node(0, self.start, 0)}
        self.edges = {}
        self.shortest_path = None

    def sample_random_config(self):
        # Sample a random configuration q_rand in the state space
        # You can sample from a uniform distribution within the state bounds
        # Return the sampled configuration
        in_collision = True
        while in_collision:
            q_rand = np.random.uniform([self.bounds['xmin'], self.bounds['ymin']],
                                       [self.bounds['xmax'], self.bounds['ymax']])
            in_collision = self.check_sample_in_obstacle(q_rand, 0.5)
        return q_rand

    def check_goal_reached(self, q):
        # Check if the goal has been reached
        # Return True if goal has been reached, False otherwise
        if euclidean_distance(q, self.goal) < 0.5:
            return True
        return False

    def check_sample_in_obstacle(self, q_rand, epsilon):
        for obstacle in self.obstacles:
            obstacle_position_2d = self.obstacles[obstacle]['position'][0], self.obstacles[obstacle]['position'][1]
            distance = euclidean_distance(q_rand, obstacle_position_2d)
            if distance < self.obstacles[obstacle]['radius'] + epsilon:
                #print("Sampled within obstacle")
                return True
        return False

    def nearest_neighbour(self, q_rand):
        # Return the nearest neighbor of q_rand from the tree
        min_distance = float('inf')
        nearest_neighbour = None
        for node in self.nodes:
            node_pos = self.nodes[node].position
            distance = euclidean_distance(node_pos, q_rand)
            if distance < min_distance:
                min_distance = distance
                nearest_neighbour = node
        return nearest_neighbour

    def check_collision_in_path(self, q1, q2):
        # Check for collisions in the path between two configurations q1 and q2
        for obstacle in self.obstacles:
            cx, cy, r = self.obstacles[obstacle]['position'][0], self.obstacles[obstacle]['position'][1], self.obstacles[obstacle]['radius']
            x1, y1, x2, y2 = q1[0], q1[1], q2[0], q2[1]

            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            A = 1 + m**2
            B = 2 * (m * c - m * cy - cx)
            C = cx**2 + cy**2 + c**2 - 2 * cy * c - r**2

            discriminant = B**2 - 4 * A * C
            if discriminant >= 0:
                x_intersects = [(-B - np.sqrt(discriminant)) / (2 * A), (-B + np.sqrt(discriminant)) / (2 * A)]
                y_intersects = [(m * x + c) for x in x_intersects]
                for x,y in zip(x_intersects, y_intersects):
                    if x1 <= x <= x2 or x2 <= x <= x1:
                        if y1 <= y <= y2 or y2 <= y <= y1:
                            #print("Generated path collides with an obstacle")
                            return True
        return False
    def path_cost(self, path):
        cost = 0
        for i in range(len(path)-1):
            q1 = self.nodes[path[i]].position
            q2 = self.nodes[path[i+1]].position
            cost += euclidean_distance(q1, q2)
        return cost
    def extend(self, q_near, q_rand):
        # Extend the tree from configuration q_near towards q_rand
        # Return the new configuration that is reached by this extension
        collision = self.check_collision_in_path(q_near.position, q_rand)
        if not collision:
            self.nodes[len(self.nodes)] = Node(len(self.nodes), q_rand, q_near.number)
        return collision

    def rewire(self, new_node_id):
        # Rewire the tree such that the cost of reaching q_new from the root is minimized
        old_path = self.compute_path(0, new_node_id)
        old_cost = self.path_cost(old_path)
        for node_id in self.nodes:
            if node_id != new_node_id:
                new_path = sorted(self.compute_path(0, node_id) + [new_node_id])
                new_cost = self.path_cost(new_path)
                # Check if rewiring to this node results in a lower cost
                if new_cost < old_cost:
                    collision_check = False
                    # Check for collisions in the new path
                    for node_index in range(len(new_path) - 1):
                        collision_check = self.check_collision_in_path(self.nodes[new_path[node_index]].position,
                                                     self.nodes[new_path[node_index + 1]].position)
                        if collision_check:
                            break  # Break out of the loop if collision is detected
                    if not collision_check:
                        # Rewire the tree
                        # Update the parent of the rewired node
                        old_cost = new_cost
                        self.nodes[new_node_id].parent = node_id
        # Update the edges
        all_nodes = list(self.nodes.keys())[::-1]
        for node in all_nodes:
            all_nodes.remove(node)
            if self.nodes[node].parent != node:
                if [self.nodes[node].parent, node] not in self.edges.values():
                    self.edges[len(self.edges)] = [self.nodes[node].parent, node]

    def compute_path(self, start_node, end_node):
        # Compute the path between two nodes
        start_not_reached = True
        path = [end_node]
        while start_not_reached:
            if end_node == start_node:
                start_not_reached = False
            else:
                end_node = self.nodes[end_node].parent
                path.append(end_node)
        return path

    def step(self):
        # Perform one iteration of the RRT algorithm
        # Return the new node added to the tree
        q_rand = self.sample_random_config()
        q_near = self.nearest_neighbour(q_rand)
        collision = self.extend(self.nodes[q_near], q_rand)
        if not collision:
            self.rewire(len(self.nodes)-1)
        return q_rand
    def run_rrt_star(self):
        # Perform the RRT algorithm for max_iter iterations
        # Return the shortest path found
        for i in range(self.max_iter):
            q_new = self.step()
            if self.check_goal_reached(q_new):
                self.shortest_path = self.compute_path(0, len(self.nodes)-1)
                return self.shortest_path
        return None

def illustrate_algorithm(rrt):
    plt.figure()
    for node in rrt.nodes:
        x,y = rrt.nodes[node].position
        plt.scatter(x, y)
        plt.text(x, y, str(node), fontsize=12, ha='right', va='bottom')  # Annotate with the node number
    for edge in rrt.edges:
        node_1 = rrt.nodes[rrt.edges[edge][0]]
        node_2 = rrt.nodes[rrt.edges[edge][1]]
        plt.plot([node_1.position[0], node_2.position[0]], [node_1.position[1], node_2.position[1]], 'm-', lw=2)
    for obstacle in rrt.obstacles:
        circle = obstacles[obstacle]
        circle_patch = patches.Circle((circle['position'][0], circle['position'][1]), circle['radius'], edgecolor='orange', facecolor='none', linewidth=2, label='Circle')
        plt.gca().add_patch(circle_patch)
    plt.show()


if __name__ == "__main__":
    obstacles = {1:{'position': [1, 1, 0], 'radius': 0.5}, 2:{'position': [2, 2, 0], 'radius': 0.5}, 3:{'position': [-2, -1.5, 0], 'radius': 0.5}}
    rrt = RRT(start=[0, 0], goal=[2.75, 2.75], bounds={'xmin': -3, 'xmax': 3, 'ymin': -3, 'ymax': 3, 'zmin': 0, 'zmax': 0}, obstacles=obstacles)
    shortest_path = rrt.run_rrt_star()
    print(f'Shortest path: {shortest_path}')
    illustrate_algorithm(rrt)
