import numpy as np
import plotly.graph_objects as go

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


class RRTStar:
    def __init__(self, start, goal, bounds, obstacles, epsilon, max_iter=100000, two_dim=False, star=True):
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.nodes = {0: Node(0, self.start, 0)}
        self.edges = {}
        self.shortest_path = None
        self.two_dim = two_dim
        self.star = star

    def sample_random_config(self):
        # Sample a random configuration q_rand in the state space
        # You can sample from a uniform distribution within the state bounds
        # Return the sampled configuration
        in_collision = True
        while in_collision:
            if self.two_dim:
                q_rand = np.random.uniform([self.bounds['xmin'], self.bounds['ymin'], 0],
                                           [self.bounds['xmax'], self.bounds['ymax'], 0])
            else:
                q_rand = np.random.uniform([self.bounds['xmin'], self.bounds['ymin'], self.bounds['zmin']],
                                           [self.bounds['xmax'], self.bounds['ymax'], self.bounds['zmax']])
            in_collision = self.check_sample_in_obstacle(q_rand)
        return q_rand

    def check_goal_reached(self, q):
        # Check if the goal has been reached
        # Return True if goal has been reached, False otherwise
        if euclidean_distance(q, self.goal) < 0.2:
            return True
        return False

    def check_sample_in_obstacle(self, q_rand):
        for obstacle in self.obstacles:
            obstacle_position_3d = (self.obstacles[obstacle]['position'][0], self.obstacles[obstacle]['position'][1],
                                    self.obstacles[obstacle]['position'][2])
            distance = euclidean_distance(q_rand, obstacle_position_3d)
            if distance < self.obstacles[obstacle]['radius'] + self.epsilon:
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
        # For explanation see: https://en.wikipedia.org/wiki/Line-sphere_intersection
        # Check for collisions in the path between two configurations q1 and q2
        for obstacle in self.obstacles:
            cx, cy, cz, r = self.obstacles[obstacle]['position'][0], self.obstacles[obstacle]['position'][1], \
            self.obstacles[obstacle]['position'][2], self.obstacles[obstacle]['radius']  + self.epsilon
            x1, y1, z1, x2, y2, z2 = q1[0], q1[1], q1[2], q2[0], q2[1], q2[2]

            # Direction vector of the line
            u = np.array([x2 - x1, y2 - y1, z2 - z1])

            # Vector from the sphere's center to the line's origin
            oc = np.array([x1 - cx, y1 - cy, z1 - cz])

            a = np.dot(u, u)
            b = 2 * np.dot(u, oc)
            c = np.dot(oc, oc) - r ** 2

            discriminant = b ** 2 - 4 * a * c

            if discriminant >= 0:
                # Calculate possible intersection distances
                d1 = (-b - np.sqrt(discriminant)) / (2 * a)
                d2 = (-b + np.sqrt(discriminant)) / (2 * a)

                # Check if the intersections are within the line segment
                if 0 <= d1 <= 1 or 0 <= d2 <= 1:
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
        if not self.star:
            # Update the edges
            all_nodes = list(self.nodes.keys())[::-1]
            for node in all_nodes:
                all_nodes.remove(node)
                if self.nodes[node].parent != node:
                    if [self.nodes[node].parent, node] not in self.edges.values():
                        self.edges[len(self.edges)] = [self.nodes[node].parent, node]
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
        if not collision and self.star == True:
            self.rewire(len(self.nodes)-1)
        return q_rand
    def full_run(self):
        # Perform the RRT algorithm for max_iter iterations
        # Return the shortest path found
        for i in range(self.max_iter):
            #print(self.nodes)
            q_new = self.step()
            if self.check_goal_reached(q_new):
                self.shortest_path = self.compute_path(0, len(self.nodes)-1)
                cost_shortest_path = self.path_cost(self.shortest_path)
                return self.shortest_path, cost_shortest_path

def illustrate_algorithm_3d(rrt):
    fig = go.Figure()

    for node in list(rrt.nodes.keys())[1:-1]:
        x, y, z = rrt.nodes[node].position
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers+text', marker=dict(size=1), textposition='bottom center'))
    fig.add_trace(go.Scatter3d(x=[rrt.start[0]], y=[rrt.start[1]], z=[rrt.start[2]], mode='markers+text', marker=dict(size=25), text=['start'], textposition='top center'))
    fig.add_trace(go.Scatter3d(x=[rrt.goal[0]], y=[rrt.goal[1]], z=[rrt.goal[2]], mode='markers+text', marker=dict(size=25), text=['goal'], textposition='top center'))
    for edge in rrt.edges:
        node_1 = rrt.nodes[rrt.edges[edge][0]]
        node_2 = rrt.nodes[rrt.edges[edge][1]]
        fig.add_trace(go.Scatter3d(x=[node_1.position[0], node_2.position[0]],
                                  y=[node_1.position[1], node_2.position[1]],
                                  z=[node_1.position[2], node_2.position[2]],
                                  mode='lines'))

    for obstacle in rrt.obstacles:
        sphere = rrt.obstacles[obstacle]
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = sphere['position'][0] + sphere['radius'] * np.cos(u) * np.sin(v)
        y = sphere['position'][1] + sphere['radius'] * np.sin(u) * np.sin(v)
        z = sphere['position'][2] + sphere['radius'] * np.cos(v)
        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.5))

    fig.show()

if __name__ == "__main__":
    obstacles = {1:{'position': [1, 1, 0], 'radius': 0.5}, 2:{'position': [2, 2, 0], 'radius': 0.5}, 3:{'position': [-2, -1.5, 0], 'radius': 0.5}}
    for i in range(12):
        obstacles[i+4] = {'position': [np.random.uniform(0, 4), np.random.uniform(0, 4), np.random.uniform(0, 3)], 'radius': 0.5}
    rrt_star = RRTStar(start=[0, 0, 0], goal=[5, 5, 3], bounds={'xmin': -8, 'xmax': 8, 'ymin': -8, 'ymax': 5.5, 'zmin': 0, 'zmax': 5}, obstacles=obstacles)
    shortest_path = rrt_star.full_run()
    print(f'Shortest path: {shortest_path}')
    illustrate_algorithm_3d(rrt_star)

