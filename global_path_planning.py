import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import permutations

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


test_node = Node(1, [0, 0], 0)
print(test_node)

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

    def check_sample_in_obstacle(self, q_rand, epsilon):
        for obstacle in self.obstacles:
            obstacle_position_2d = self.obstacles[obstacle]['position'][0], self.obstacles[obstacle]['position'][1]
            distance = euclidean_distance(q_rand, obstacle_position_2d)
            if distance < self.obstacles[obstacle]['radius'] + epsilon:
                print("Sampled within obstacle")
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
            if discriminant >= 0:
                x_intersects = [(-B - np.sqrt(discriminant)) / (2 * A), (-B + np.sqrt(discriminant)) / (2 * A)]
                y_intersects = [(m * x + c) for x in x_intersects]
                for x,y in zip(x_intersects, y_intersects):
                    if x1 <= x <= x2 or x2 <= x <= x1:
                        if y1 <= y <= y2 or y2 <= y <= y1:
                            print("Generated path collides with an obstacle")
                            return True
        return False

    def extend(self, q_near, q_rand):
        # Extend the tree from configuration q_near towards q_rand
        # Return the new configuration that is reached by this extension
        collision = self.check_collision_in_path(q_near.position, q_rand)
        if not collision:

            self.nodes[len(self.nodes)] = Node(len(self.nodes), q_rand, q_near.number)
            #self.edges[len(self.edges)] = [q_near.number, (len(self.nodes)-1)]
            print(f'nodes: {self.nodes}')
            print(f'edges: {self.edges}')
        return collision
    def tree_cost(self, tree):
        # Compute the cost of the current tree
        cost = 0
        for edge in tree:
            q1 = self.nodes[self.edges[edge][0]].position
            q2 = self.nodes[self.edges[edge][1]].position
            cost += euclidean_distance(q1, q2)
        return cost

    def path_cost(self, path):
        cost = 0
        print(f'path in compute path: {path}')
        for i in range(len(path)-1):
            q1 = self.nodes[path[i]].position
            q2 = self.nodes[path[i+1]].position
            print(f'q1: {q1}')
            print(f'q2: {q2}')
            cost += euclidean_distance(q1, q2)
        return cost


    def rewire(self, new_node_id):
        # Rewire the tree such that the cost of reaching q_new from the root is minimized
        # You can use the tree_cost function to compute the cost of the tree

        print(f'new_node_id: {new_node_id}')
        old_path = self.compute_path(0, new_node_id)
        old_cost = self.path_cost(old_path)
        print(f'old_path: {old_path}')
        print(f'old_cost: {old_cost}')

        for node_id in self.nodes:
            if node_id != new_node_id:
                print(f'node_id: {node_id}')
                new_path = sorted(self.compute_path(0, node_id) + [new_node_id])
                new_cost = self.path_cost(new_path)
                print(f'new_path: {new_path}')
                print(f'new_cost: {new_cost}')

                # Check if rewiring to this node results in a lower cost
                if new_cost < old_cost:
                    collision_check = False
                    for node_index in range(len(new_path) - 1):
                        collision_check = self.check_collision_in_path(self.nodes[new_path[node_index]].position,
                                                                       self.nodes[new_path[node_index + 1]].position)
                        if collision_check:
                            break  # Break out of the loop if collision is detected

                    if not collision_check:
                        # Rewire the tree
                        parent_id = self.nodes[node_id].parent
                        parent_position = self.nodes[parent_id].position
                        new_edge = [parent_id, new_node_id]
                        old_cost = new_cost

                        # Update the parent of the rewired node
                        self.nodes[new_node_id].parent = node_id
                        # Update edges
        all_nodes = list(self.nodes.keys())[::-1]
        print(f'all_nodes: {all_nodes}')
        for node in all_nodes:
            all_nodes.remove(node)
            if self.nodes[node].parent != node:
                if [self.nodes[node].parent, node] not in self.edges.values():
                    self.edges[len(self.edges)] = [self.nodes[node].parent, node]
        print(f'edges after rewiring: {self.edges}')

    def check_goal_reached(self, q):
        # Check if the goal has been reached
        # Return True if goal has been reached, False otherwise
        if euclidean_distance(q, self.goal) < 0.5:
            return True
        return False

    def step(self):
        # Perform one iteration of the RRT algorithm
        # Return the new node added to the tree
        goal_reached = False
        while not goal_reached:
            q_rand = self.sample_random_config()
            print(f'the random sample is {q_rand}')
            q_near = self.nearest_neighbour(q_rand)
            print(f'the nearest neighbour is {q_near}')
            collision = self.extend(self.nodes[q_near], q_rand)
            if not collision:
                goal_reached = self.check_goal_reached(q_rand)
                self.rewire(len(self.nodes)-1)
        print(f'nodes: {self.nodes}')
        return q_rand

    def give_shortest_path(self):
        start_not_reached = True
        goal_parent = self.nodes[len(self.nodes)-1].parent
        path = [len(self.nodes)-1, goal_parent]
        while start_not_reached:
            if goal_parent == 0:
                start_not_reached = False
            else:
                goal_parent = self.nodes[goal_parent].parent
                path.append(goal_parent)
        return path

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

obstacles = {1:{'position': [1, 1, 0], 'radius': 0.5}}
test_rrt = RRT(start=[0, 0], goal=[8, 4], bounds={'xmin': 0, 'xmax': 16, 'ymin': 0, 'ymax': 16, 'zmin': 0, 'zmax': 0}, obstacles=obstacles)

test_rrt.step()

plt.figure()
previous_node = False
print(test_rrt.give_shortest_path())
for node in test_rrt.nodes:
    x,y = test_rrt.nodes[node].position
    plt.scatter(x, y)
    plt.text(x, y, str(node), fontsize=12, ha='right', va='bottom')  # Annotate with the node number
for edge in test_rrt.edges:
    node_1 = test_rrt.nodes[test_rrt.edges[edge][0]]
    node_2 = test_rrt.nodes[test_rrt.edges[edge][1]]
    plt.plot([node_1.position[0], node_2.position[0]], [node_1.position[1], node_2.position[1]], 'm-', lw=2)
circle = (1, 1, 0.5)

circle_patch = patches.Circle((circle[0], circle[1]), circle[2], edgecolor='orange', facecolor='none', linewidth=2, label='Circle')
plt.gca().add_patch(circle_patch)
plt.show()