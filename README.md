# PDM-group-22

## Getting started
This readme explains how to install and run a RRT* planning algorithm for a mobile manipulator. Created for the course Planning & Decision Making by group 22

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

## Running the Main Code

To run the main code from the terminal, you can use the following command:

```bash
python main.py --render --hard --rrt_star
'''
### Command-Line Options:
--render: Enable visualization during the execution of the algorithm.

--hard: Set this flag to make the goal position harder. If omitted, the default is an easier goal position.

--rrt_star: Use the RRT* algorithm for global path planning. If omitted, the default is to use the algorithm without RRT*.
