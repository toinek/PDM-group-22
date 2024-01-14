from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
import numpy as np

spawn_rotation = 0.5 * np.pi

class ObstacleAdder():
    def __init__(self, env):
        self.env = env
        self.spheres = {1: {'pos': [1.5, 0, 0], 'radius' : 0.2}, 2: {'pos': [1.5, 2, 0], 'radius' : 0.2},
                        3: {'pos': [3.5, 2, 0], 'radius' : 0.2}, 4: {'pos': [3.5, 4.5, 0], 'radius' : 0.2}, 5: {'pos': [6, 7, 0], 'radius' : 0.2}}

    def add_spheres(self):
        for sphere in self.spheres:
            pos = self.spheres[sphere]['pos']
            radius = self.spheres[sphere]['radius']
            sphere_obst_dict = {
                "type": "sphere",
                'movable': False,
                "geometry": {"position": pos, "radius": radius},
            }
            sphere_obst = SphereObstacle(name=f'obstacle_{pos[0]}_{pos[1]}_{pos[2]}', content_dict=sphere_obst_dict)
            self.env.add_obstacle(sphere_obst)


    def add_walls(self, boundary):
        wall_length = boundary
        wall_obstacles_dicts = [
            {
                'type': 'box',
                'geometry': {
                    'position': [wall_length / 2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': 0.1
                },
                'high': {
                    'position': [wall_length / 2.0, 0.0, 0.4],
                    'width': wall_length,
                    'height': 0.8,
                    'length': 0.1,
                },
                'low': {
                    'position': [wall_length / 2.0, 0.0, 0.4],
                    'width': wall_length,
                    'height': 0.8,
                    'length': 0.1,
                },
            },
            {
                'type': 'box',
                'geometry': {
                    'position': [0.0, wall_length / 2.0, 0.4], 'width': 0.1, 'height': 0.8, 'length': wall_length
                },
                'high': {
                    'position': [0.0, wall_length / 2.0, 0.4],
                    'width': 0.1,
                    'height': 0.8,
                    'length': wall_length,
                },
                'low': {
                    'position': [0.0, wall_length / 2.0, 0.4],
                    'width': 0.1,
                    'height': 0.8,
                    'length': wall_length,
                },
            },
            {
                'type': 'box',
                'geometry': {
                    'position': [0.0, -wall_length / 2.0, 0.4], 'width': 0.1, 'height': 0.8, 'length': wall_length
                },
                'high': {
                    'position': [0.0, -wall_length / 2.0, 0.4],
                    'width': 0.1,
                    'height': 0.8,
                    'length': wall_length,
                },
                'low': {
                    'position': [0.0, -wall_length / 2.0, 0.4],
                    'width': 0.1,
                    'height': 0.8,
                    'length': wall_length,
                },
            },
            {
                'type': 'box',
                'geometry': {
                    'position': [-wall_length / 2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': 0.1
                },
                'high': {
                    'position': [-wall_length / 2.0, 0.0, 0.4],
                    'width': wall_length,
                    'height': 0.8,
                    'length': 0.1,
                },
                'low': {
                    'position': [-wall_length / 2.0, 0.0, 0.4],
                    'width': wall_length,
                    'height': 0.8,
                    'length': 0.1,
                },
            },
        ]

        # add walls to the environment
        wall_obstacles = [BoxObstacle(name=f"wall_{i}", content_dict=obst_dict) for i, obst_dict in
                          enumerate(wall_obstacles_dicts)]
        for i in range(len(wall_obstacles)):
            self.env.add_obstacle(wall_obstacles[i])