# PDM-group-22
This repository was made for the group assignment of the RO47005 Planning & Decision Making course at the TU Delft. In this project a RRT and RRT* algorithm have been compared as motion planners for a mobile manipulator. 

## Getting started
This readme explains how to install and run the planning algorithm in the created environment
This file will contain the following contents:
- Repository contents
- Pre-requisities
- Setup
- Run guide

## Repository Content

The repository contains the following files:

- **main.py:** The main script that orchestrates the execution of the robotic motion planning and decision-making.

- **add_obstacles.py:** Python script responsible for adding obstacles to the environment.

- **kinematics.py:** Module containing the kinematics for the simplified robotic arm.

- **global_path_planning_3d.py:** The RRT/RRT* global path planning algorithm suitable for 3D environments.

- **local_arm_control.py:** Implementation of the PID controller for local arm control.

- **local_path_planning.py:** Implementation of the PID controller as local path planner.

- **README.md:** This file providing instructions, information, and documentation about the project.

- **requirements.txt:** File specifying the dependencies required to run the project.

Feel free to explore and modify these files based on your project requirements and preferences.

## Pre-requisites
+ Python >= 3.8
+ pip3
+ git

## Setup

1. **Clone the repository:**

```bash
git clone git@github.com:toinek/PDM-group-22.git
cd PDM-group-22
```

2. **Install additional dependencies:**

```bash
pip install -r requirements.txt
```

## Run guide
To run the main code from within a terminal, you can use the following command:

```bash
python main.py --render --hard --rrt_star
```

### Command-Line Options:
--render: Enable visualization during the execution of the algorithm.

--hard: Set this flag to make the goal position harder. If omitted, the default is an easier goal position.

--rrt_star: Use the RRT* algorithm for global path planning. If omitted, the default is to use the RRT algorithm.
