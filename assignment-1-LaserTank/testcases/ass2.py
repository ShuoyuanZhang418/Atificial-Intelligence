import copy
import datetime
import sys
import time

import numpy as np

from angle import Angle
from problem_spec import ProblemSpec
from robot_config import write_robot_config_list_to_file
from robot_config import *
from tester import test_config_equality
from tester import *

"""
Template file for you to implement your solution to Assignment 2. Contains a class you can use to represent graph nodes,
and a method for finding a path in a graph made up of GraphNode objects.

COMP3702 2020 Assignment 2 Support Code
"""


class GraphNode:
    """
    Class representing a node in the state graph. You should create an instance of this class each time you generate
    a sample.
    """

    def __init__(self, spec, config):
        """
        Create a new graph node object for the given config.

        Neighbors should be added by appending to self.neighbors after creating each new GraphNode.

        :param spec: ProblemSpec object
        :param config: the RobotConfig object to be stored in this node
        """
        self.spec = spec
        self.config = config
        self.config_list = []

    def __eq__(self, config1, config2):
        return test_config_equality(config1, config2, self.spec)

    def __hash__(self, config):
        return hash(tuple(config.points))

    def get_successors(self):
        return self.neighbors

    @staticmethod
    def add_connection(n1, n2):
        """
        Creates a neighbor connection between the 2 given GraphNode objects.

        :param n1: a GraphNode object
        :param n2: a GraphNode object
        """
        n1.neighbors.append(n2)
        n2.neighbors.append(n1)

    def generate_new_random_config(self, config):
        if config.ee1_grappled:
            ee1x, ee1y = config.get_ee1()
        else:
            ee2x, ee2y = config.get_ee2()
        # generate ee1_length
        config_lengths = []
        for i in range(self.spec.num_segments):
            random_length = np.random.uniform(self.spec.min_lengths[i], self.spec.max_lengths[i])
            config_lengths.append(random_length)
        # generate ee1_angles
        config_angles = []
        random_degree = np.random.randint(-165, 165, self.spec.num_segments)
        for i in range(self.spec.num_segments):
            random_angle = Angle(degrees=random_degree[i])
            config_angles.append(random_angle)
        # generate config
        if config.ee1_grappled:
            random_config = make_robot_config_from_ee1(ee1x, ee1y, config_angles, config_lengths, ee1_grappled=True)
        else:
            random_config = make_robot_config_from_ee2(ee2x, ee2y, config_angles, config_lengths, ee2_grappled=True)
        return random_config

    def generate_random_config(self, config):
        """random generate a valid state (RobotConfig) based on the based_config.
        checking rules : (1)angle_constraints (2)collide with obstacle (3)collide with itself (4) out of environment bounds"""
        random_config = self.generate_new_random_config(config)
        while not (self.valid_config(random_config)):
            random_config = self.generate_new_random_config(config)
        return random_config

    def valid_config(self, config):
        """check whether a config is valid or not.
        valid config : return(True)
        invalid config: return(False)
        # check whether the random meet requirement
        # (1) test_angle_constraints
        # (2) not collide with obstacle
        # (3) not collide with itself
        # (4) not out of the environment bounds : """
        # spec = ProblemSpec(input_file)
        return (test_angle_constraints(config, self.spec)
                and test_obstacle_collision(config, self.spec, self.spec.obstacles)
                and test_self_collision(config, self.spec)
                and test_environment_bounds(config))

    def find_gap_in_between_obstacles(self):
        bunch_config = {}
        i = 0
        j = 0
        gaps = set()
        if len(self.spec.obstacles) < 2:
            return gaps
        for i in range(len(self.spec.obstacles)):
            for j in range(len(self.spec.obstacles)):
                if i >= j:
                    continue
                else:
                    if self.spec.obstacles[i].x2 <= self.spec.obstacles[j].x1:
                        x_from = self.spec.obstacles[i].x2
                        x_to = self.spec.obstacles[j].x1
                    elif self.spec.obstacles[i].x1 >= self.spec.obstacles[j].x2:
                        x_from = self.spec.obstacles[j].x2
                        x_to = self.spec.obstacles[i].x1
                    else:
                        x_from = max(self.spec.obstacles[i].x1, self.spec.obstacles[j].x1)
                        x_to = min(self.spec.obstacles[i].x2, self.spec.obstacles[j].x2)
                    if self.spec.obstacles[i].y2 <= self.spec.obstacles[j].y1:
                        y_from = self.spec.obstacles[i].y2
                        y_to = self.spec.obstacles[j].y1
                    elif self.spec.obstacles[i].y1 >= self.spec.obstacles[j].y2:
                        y_from = self.spec.obstacles[j].y2
                        y_to = self.spec.obstacles[i].y1
                    else:
                        y_from = max(self.spec.obstacles[i].y1, self.spec.obstacles[j].y1)
                        y_to = min(self.spec.obstacles[i].y2, self.spec.obstacles[j].y2)
            gaps.add((x_from, x_to, y_from, y_to))
        return gaps

    def generate_bunch_config_in_between_obstacles(self, based_config, goal_config, n):
        gaps = self.find_gap_in_between_obstacles()
        bunch_config = {}
        i = 0
        while i < n:
            if len(gaps) == 0:
                config = self.generate_random_config(based_config)
                bunch_config[config] = config
                i = i + 1
            else:
                for gap in gaps:
                    config = self.generate_random_config(based_config)
                    point = self.get_config_non_grappled_ee_point(config)
                    if gap[0] <= point[0] <= gap[1] and gap[2] <= point[1] <= gap[3]:
                        bunch_config[config] = config
                        i = i + 1
        return bunch_config

    def get_config_non_grappled_ee_point(self, config: RobotConfig):
        points_count = len(config.points)
        if config.ee1_grappled is True:
            return config.points[points_count - 1]

        if config.ee2_grappled is True:
            return config.points[0]

    def get_config_grappled_ee_point(self, config: RobotConfig):
        points_count = len(config.points)
        if config.ee1_grappled is True:
            return config.points[0]

        if config.ee2_grappled is True:
            return config.points[points_count - 1]

    def get_config_non_grappled_ee_arm_length(self, config: RobotConfig):
        points_count = len(config.points)
        arm_count = points_count - 1
        if config.ee1_grappled is True:
            return config.lengths[arm_count - 1]

        if config.ee2_grappled is True:
            return config.lengths[0]

    def generate_random_config_near_goal(self, goal_config, n):
        ee1x, ee1y = goal_config.get_ee1()
        ee2x, ee2y = goal_config.get_ee2()
        # generate ee1_length
        bunch_config = {}
        j = 0
        while j < n:
            config_lengths = []
            for i in range(self.spec.num_segments):
                if goal_config.lengths[i] == self.spec.max_lengths[i]:
                    random_length = goal_config.lengths[i] - 0.002 * j
                elif goal_config.lengths[i] == self.spec.min_lengths[i]:
                    random_length = goal_config.lengths[i] + 0.002 * j
                config_lengths.append(random_length)
            # generate ee1_angles
            config_angles = []
            for i in range(self.spec.num_segments):
                if j < n/2:
                    if goal_config.ee1_grappled:
                        random_angle = goal_config.ee1_angles[i] - 0.02 * j
                    else:
                        random_angle = goal_config.ee2_angles[i] - 0.02 * j
                    config_angles.append(random_angle)
                else:
                    if goal_config.ee1_grappled:
                        random_angle = goal_config.ee1_angles[i] + 0.02 * (j + 1 - n/2)
                    else:
                        random_angle = goal_config.ee2_angles[i] + 0.02 * (j + 1 - n/2)
                    config_angles.append(random_angle)
            # generate config
            if goal_config.ee1_grappled:
                random_config = make_robot_config_from_ee1(ee1x, ee1y, config_angles, config_lengths, ee1_grappled=True)
            else:
                random_config = make_robot_config_from_ee2(ee2x, ee2y, config_angles, config_lengths, ee2_grappled=True)
            if self.valid_config(random_config):
                bunch_config[random_config] = random_config
            j = j + 1
        return bunch_config

    def generate_double_ee_config(self, config, eex, eey, n):
        random_config = self.generate_new_random_config(config)
        bunch_config = {}
        i = 0
        while i < n:
            if config.ee1_grappled:
                while not ((eex, eey) == random_config.get_ee2()):
                    random_config = self.generate_new_random_config(config)
            else:
                while not ((eex, eey) == random_config.get_ee1()):
                    random_config = self.generate_new_random_config(config)
            bunch_config[random_config] = random_config
            i = i + 1
        return bunch_config

    def generate_bunch_config(self, based_config,n):
        bunch_config = {}
        i = 0
        while i < n:
            config = self.generate_random_config(based_config)
            bunch_config[config] = config
            i = i + 1

        return bunch_config

    def generate_bunch_config_outside_obstacles(self, based_config, goal_config, n, direction):
        """random generate n configs
        return: config dictionary"""
        # print('Points outside obstacles')
        gaps = self.find_gap_in_between_obstacles()
        bunch_config = {}
        i = 0
        while i < n:
            if len(gaps) == 0:
                config = self.generate_random_config(based_config)
                bunch_config[config] = config
                i = i + 1
            else:
                for gap in gaps:
                    config = self.generate_random_config(based_config)
                    point = self.get_config_non_grappled_ee_point(config)
                    if direction == 'X':
                        if point[0] <= gap[0] or point[0] >= gap[1]:
                            bunch_config[config] = config
                            i = i + 1
                    if direction == 'Y':
                        if point[1] <= gap[2] or point[1] >= gap[3]:
                            bunch_config[config] = config
                            i = i + 1

        return bunch_config

    def generate_config_vector(self, config):
        """generate config vector for the given config.
        config: config object
        return : [Angle,...,length,...]"""
        config_vector = []
        if self.spec.initial.ee1_grappled:
            config_vector.extend(config.ee1_angles)
            config_vector.extend(config.lengths)
        else:
            config_vector.extend(config.ee2_angles)
            config_vector.extend(config.lengths)
        return config_vector
    """
    def calculate_config_distance(self, config1: RobotConfig, config2: RobotConfig):
        vector1 = self.generate_config_vector(config1)
        vector2 = self.generate_config_vector(config2)
        distance = 0
        for i in range(len(vector1)):
            if isinstance(vector1[i], Angle):
                angle1 = vector1[i]
                angle1_radians = angle1.in_radians()
                angle2 = vector2[i]
                angle2_radians = angle2.in_radians()
                distance = distance + (angle1_radians - angle2_radians)*(angle1_radians - angle2_radians)
            else:
                distance = distance + (vector1[i] - vector2[i]) * (vector1[i] - vector2[i])
        distance = math.sqrt(distance)
        return distance
"""

    def calculate_config_distance(self, config1: RobotConfig, config2: RobotConfig):
        points1 = config1.points
        points2 = config2.points
        distance = 0
        for i in range(self.spec.num_segments + 1):
            distance += np.sqrt(np.square(points1[i][0] - points2[i][0]) + np.square(points1[i][1] - points2[i][1]))
        return distance

    def collision_free(self, config1, config2):
        config_new = copy.deepcopy(config1)
        #config_new = RobotConfig([i for i in config1.lengths], ee1x=config1.get_ee1()[0], ee1y=config1.get_ee1()[1],
                                         #ee1_angles=[i for i in config1.ee1_angles],
                                         #ee1_grappled=True)

        diff = np.array(self.generate_config_vector(config_new)) - np.array(
            self.generate_config_vector(config2))  # [angle,angle,..l,l,..]

        angles_diff = diff[0:self.spec.num_segments]
        angle_preminum = Angle(radians=0.02)
        steps = [item.in_radians() / angle_preminum.in_radians() for item in angles_diff]  # steps : list

        length_preminum = 0.02
        length_steps = diff[self.spec.num_segments:] / length_preminum
        steps.extend(length_steps)  # steps: [0.0, -523.5987755983568, 0.0, 0.0, -499.99999999999994, 0.0]


        while not (test_config_equality(config_new, config2, self.spec)):
            if config_new.ee1_grappled:
                # change the angels
                for i in range(self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.ee1_angles[i] += np.sign(steps[i]) * (-1) * angle_preminum
                        # update the steps
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.ee1_angles[i] = config2.ee1_angles[i]
                        # update the steps
                        steps[i] = 0
                for i in range(self.spec.num_segments, 2 * self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.lengths[i - self.spec.num_segments] += np.sign(steps[i]) * (-1) * length_preminum
                        # update the steps
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.lengths[i - self.spec.num_segments] = config2.lengths[i - self.spec.num_segments]
                        # update the steps
                        steps[i] = 0

                config_new = RobotConfig(config_new.lengths, ee1x=config_new.get_ee1()[0], ee1y=config_new.get_ee1()[1],
                                         ee1_angles=config_new.ee1_angles,
                                         ee1_grappled=True)

            else:
                # change the angels
                for i in range(self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.ee2_angles[i] += np.sign(steps[i]) * (-1) * angle_preminum
                        # update the steps
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.ee2_angles[i] = config2.ee2_angles[i]
                        # update the steps
                        steps[i] = 0
                for i in range(self.spec.num_segments, 2 * self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.lengths[i - self.spec.num_segments] += np.sign(steps[i]) * (-1) * length_preminum
                        # update the steps
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.lengths[i - self.spec.num_segments] = config2.lengths[i - self.spec.num_segments]
                        # update the steps
                        steps[i] = 0

                config_new = RobotConfig(config_new.lengths, ee2x=config_new.get_ee2()[0], ee2y=config_new.get_ee2()[1],
                                         ee2_angles=config_new.ee2_angles,
                                         ee2_grappled=True)

            # check collision free
            if not (self.valid_config(config_new)):
                return False
        return True

    def generate_q_new_config(self, p: RobotConfig, q: RobotConfig, delta_q):
        p_vector = self.generate_config_vector(p)
        q_vector = self.generate_config_vector(q)
        q_new_vector = []
        for i in range(len(p_vector)):
            if isinstance(p_vector[i], Angle):
                radians_val = (q_vector[i].in_radians() - p_vector[i].in_radians())*delta_q + p_vector[i].in_radians()
                angle = Angle(radians=radians_val)
                q_new_vector.append(angle)
            else:
                q_new_vector.append((q_vector[i] - p_vector[i])*delta_q + p_vector[i])

        config_angles = []
        for i in range(len(p.ee1_angles)):
            config_angles.append(q_new_vector[i])

        config_lengths = []
        for i in range(len(p.ee1_angles), len(p.ee1_angles) + len(p.lengths)):
            config_lengths.append(q_new_vector[i])

        if q.ee1_grappled:
            random_config = make_robot_config_from_ee1(q.points[0][0], q.points[0][1], config_angles, config_lengths, ee1_grappled = True)
        else:
            random_config = make_robot_config_from_ee2(q.points[-1][0], q.points[-1][1], config_angles, config_lengths, ee2_grappled = True)

        return random_config

    def prm(self, initial_config, goal_config, n):
        """generate a dictionary to generate out graph with configs
        input: bunch_configs  [config, config, config,...]
        output: {config:[config,config], ...}"""
        bunch_configs = {}
        if self.spec.num_grapple_points < 2:
            bunch_configs_random = self.generate_bunch_config(initial_config, n)
            bunch_configs.update(bunch_configs_random)
            #bunch_configs_x = self.generate_bunch_config_outside_obstacles(initial_config, goal_config, n, 'X')
            #bunch_configs_y = self.generate_bunch_config_outside_obstacles(initial_config, goal_config, n, 'Y')
            #bunch_configs_near_obstacles = self.generate_bunch_config_in_between_obstacles(initial_config, goal_config, n)
            #bunch_configs.update(bunch_configs_x)
            #bunch_configs.update(bunch_configs_y)
            #bunch_configs.update(bunch_configs_near_obstacles)
        elif self.spec.num_grapple_points == 6:
            bunch_configs_random = self.generate_bunch_config(initial_config, 15*n)
            bunch_configs.update(bunch_configs_random)
            a = self.spec.grapple_points[1][0]
            b = self.spec.grapple_points[1][1]
            config2 = make_robot_config_from_ee2(a, b, self.spec.initial.ee1_angles, self.spec.initial.lengths, ee2_grappled=True)
            bunch_configs_random2 = self.generate_bunch_config(config2, 15*n)
            bunch_configs.update(bunch_configs_random2)
            #double_ee_config = self.generate_double_ee_config(initial_config, a, b, n)
            #bunch_configs.update(double_ee_config)
        elif self.spec.num_grapple_points == 7:
            bunch_configs_random = self.generate_bunch_config(initial_config, 10 * n)
            bunch_configs.update(bunch_configs_random)
            config2 = make_robot_config_from_ee2(self.spec.grapple_points[1][0], self.spec.grapple_points[1][1],
                                                 self.spec.initial.ee1_angles,self.spec.initial.lengths, ee2_grappled=True)
            bunch_configs_random2 = self.generate_bunch_config(config2, 10 * n)
            bunch_configs.update(bunch_configs_random2)
            config3 = make_robot_config_from_ee1(self.spec.grapple_points[2][0], self.spec.grapple_points[2][1],
                                                 self.spec.initial.ee1_angles,self.spec.initial.lengths, ee1_grappled=True)
            bunch_configs_random3 = self.generate_bunch_config(config3, 10 * n)
            bunch_configs.update(bunch_configs_random3)
        #bunch_configs_previous = {}
        #for k, v in initial_graph.items():
            #bunch_configs_previous[k] = k

        #bunch_configs.update(bunch_configs_previous)

        # add the initial and goal configs
        #if len(initial_graph) == 0:
        bunch_configs[initial_config] = initial_config
        bunch_configs[goal_config] = goal_config
        bunch_configs_near_goal = self.generate_random_config_near_goal(goal_config, n/5)
        bunch_configs.update(bunch_configs_near_goal)
        # calculate distance between each pair of config
        distance_list = []
        for config in bunch_configs:
            for config_another in bunch_configs:
                if test_config_equality(config, config_another, self.spec):
                    distance = 0
                else:
                    distance = self.calculate_config_distance(config, config_another)
                    distance_list.append(distance)
        distance_list.sort(reverse=False)
        threshold_distance = distance_list[int(0.1 * len(distance_list))]
        print(threshold_distance)

        graph = {}
        #graph.update(initial_graph)
        for config in bunch_configs:
            if config not in graph:
                graph[config] = []
            for config_another, v in bunch_configs.items():
                distance = self.calculate_config_distance(config, config_another)
                if distance == 0 or test_config_equality(config, config_another, self.spec):
                    continue
                if 0 < distance <= 0.5:
                    if self.collision_free(config, config_another):
                        graph[config].append(config_another)
                    else:
                        q_new_config = self.generate_q_new_config(config, config_another, 0.3)
                        distance = self.calculate_config_distance(config, q_new_config)
                        if distance <= 0.8 and self.collision_free(config, q_new_config):
                            graph[config].append(q_new_config)

        return graph

    def steps_generator(self, config1, config2):
        # new
        """generate the moving steps between two collission-free configs.
        input: two configs (need collision-free)
        return: return a list of configs with slight move.
        note: result list will not include the config2. """

        result_list = []
        config_new = copy.deepcopy(config1)
        diff = np.array(self.generate_config_vector(config_new)) - np.array(
            self.generate_config_vector(config2))  # [angle,angle,..l,l,..]
        angles_diff = diff[0:self.spec.num_segments]
        angle_preminum = Angle(radians=0.001)
        steps = [item.in_radians() / angle_preminum.in_radians() for item in angles_diff]  # steps : list
        length_preminum = 0.001
        length_steps = diff[self.spec.num_segments:] / length_preminum
        steps.extend(length_steps)  # steps: [0.0, -523.5987755983568, 0.0, 0.0, -499.99999999999994, 0.0]
        # max_index = list(np.abs(steps)).index(max(np.abs(steps)))  # max_index : 1

        # add the first element
        config_first = copy.deepcopy(config1)
        result_list.append(config_first)
        while not (test_config_equality(config_new, config2, self.spec)):
            if config_new.ee1_grappled:
                for i in range(self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.ee1_angles[i] += np.sign(steps[i]) * (-1) * angle_preminum
                        # update the steps
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.ee1_angles[i] = config2.ee1_angles[i]
                        # update the steps
                        steps[i] = 0
                for i in range(self.spec.num_segments, 2 * self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.lengths[i - self.spec.num_segments] += np.sign(steps[i]) * (-1) * length_preminum
                        # update the steps
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.lengths[i - self.spec.num_segments] = config2.lengths[i - self.spec.num_segments]
                        # update the steps
                        steps[i] = 0
                # change the angels
                # if max_index < self.num_segments:
                # if np.abs(steps[max_index]) > 1:
                # config_new.ee1_angles[max_index] += np.sign(steps[max_index]) * (-1) * angle_preminum
                # update the steps
                # steps[max_index] += np.sign(steps[max_index]) * (-1)
                # else:
                # config_new.ee1_angles[max_index] = config2.ee1_angles[max_index]
                # update the steps
                # steps[max_index] = 0
                # change the lengths
                # else:
                # if np.abs(steps[max_index]) > 1:
                # config_new.lengths[max_index - self.num_segments] += np.sign(steps[max_index]) * (
                # -1) * length_preminum
                # update the steps
                # steps[max_index] += np.sign(steps[max_index]) * (-1)
                # else:
                # config_new.lengths[max_index - self.num_segments] = config2.lengths[
                # max_index - self.num_segments]
                # update the steps
                # steps[max_index] = 0
                # update the points
                config_new = RobotConfig(config_new.lengths, ee1x=config_new.get_ee1()[0], ee1y=config_new.get_ee1()[1],
                                         ee1_angles=config_new.ee1_angles,
                                         ee1_grappled=True)
                # print("0:", np.array(config_new.ee1_angles[0].in_degrees()))
                # print("1:", np.array(config_new.ee1_angles[1].in_degrees()))
                # print("2:", np.array(config_new.lengths[0]))
                # print("points:",config_new.points)

            else:
                for i in range(self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.ee2_angles[i] += np.sign(steps[i]) * (-1) * angle_preminum
                        # update the steps
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.ee2_angles[i] = config2.ee1_angles[i]
                        # update the steps
                        steps[i] = 0
                for i in range(self.spec.num_segments, 2 * self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.lengths[i - self.spec.num_segments] += np.sign(steps[i]) * (-1) * length_preminum
                        # update the steps
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.lengths[i - self.spec.num_segments] = config2.lengths[i - self.spec.num_segments]
                        # update the steps
                        steps[i] = 0
                # change the angels
                # if max_index < self.num_segments:
                # if np.abs(steps[max_index]) > 1:
                # config_new.ee2_angles[max_index] += np.sign(steps[max_index]) * (-1) * angle_preminum
                # update the steps
                # steps[max_index] += np.sign(steps[max_index]) * (-1)
                # else:
                # config_new.ee2_angles[max_index] = config2.ee2_angles[max_index]
                # update the steps
                # steps[max_index] = 0
                # change the lengths
                # else:
                # if np.abs(steps[max_index]) > 1:
                # config_new.lengths[max_index - self.num_segments] += np.sign(steps[max_index]) * (
                # -1) * length_preminum
                # update the steps
                # steps[max_index] += np.sign(steps[max_index]) * (-1)
                # else:
                # config_new.lengths[max_index - self.num_segments] = config2.lengths[
                # max_index - self.num_segments]
                # update the steps
                # steps[max_index] = 0

                # update the points
                config_new = RobotConfig(config_new.lengths, ee2x=config_new.get_ee1()[0], ee2y=config_new.get_ee1()[1],
                                         ee2_angles=config_new.ee1_angles,
                                         ee2_grappled=True)

            # update the max_index
            # max_index = list(np.abs(steps)).index(max(np.abs(steps)))

            # update the result_list
            config_record = copy.deepcopy(config_new)
            result_list.append(config_record)
        return result_list

    def steps(self, config_list):
        # new
        """output the result as a list of RobotConfig. At the same time, print it out on console.
        input: a list of configs [config1, config2, config3] (config is on the graph)
        output : [RobotConfig, RobotConfig, RobotConfig,...]
        output_file :  output.txt """
        steps = []
        for i in range(len(config_list) - 1):
            result_list = self.steps_generator(config_list[i], config_list[i + 1])
            steps.extend(result_list)
        return steps

    def find_graph_path(self, graph):
        """
        This method performs a breadth first search of the state graph and return a list of configs which form a path
        through the state graph between the initial and the goal. Note that this path will not satisfy the primitive step
        requirement - you will need to interpolate between the configs in the returned list.

        You may use this method in your solver if you wish, or can implement your own graph search algorithm to improve
        performance.

        :param graph: all possible configs with neighbors within a specified distance
        :return: List of configs forming a path through the graph from initial to goal
        """
        # the set to store explored config
        explored = set()
        initial = self.spec.initial
        goal_config = self.spec.goal
        # the list to store fringe config
        fringe_list = [[initial]]

        while len(fringe_list) > 0:
            # pop first node in queue
            nodes = fringe_list.pop(0)
            fringe_node = nodes[-1]
            if self.__hash__(fringe_node) not in explored:
                # add node to explored list
                explored.add(self.__hash__(fringe_node))
                # if the current node has children
                if fringe_node in graph.keys():
                    for child in graph[fringe_node]:
                        solution = copy.deepcopy(nodes)
                        solution.append(child)
                        if self.__eq__(child, goal_config):
                            solution_found = True
                            return solution, solution_found
                        else:
                            fringe_list.append(solution)
        solution_found = False
        return [], solution_found


def main(arglist):
    input_file = arglist[0]
    spec = ProblemSpec(input_file)
    config = spec.initial
    solver = GraphNode(spec, config)
    if len(arglist) > 1:
        output_file = arglist[1]
    initial_config = spec.initial
    goal_config = spec.goal
    solution_found = False
    i = 0

    start = time.process_time()
    while not solution_found and i < 3:
        print("PRM round", i + 1, "start")
        graph = solver.prm(initial_config, goal_config, 200)
        print("Path find round", i + 1, "start")
        solution, solution_found = solver.find_graph_path(graph)
        i += 1

    solver.config_list.extend(solution)
    elapsed = (time.process_time() - start)
    print("Time used:", elapsed)
    # print("Solution found at", datetime.datetime.now())
    # found eventually for every leg
    if solution_found or i == 3:
        steps = solver.steps(solver.config_list)
        write_robot_config_list_to_file(output_file, steps)
    else:
        print("Solution not found")
    #input_file = arglist[0]
    #output_file = arglist[1]

    #spec = ProblemSpec(input_file)

    #init_node = GraphNode(spec, spec.initial)
    #goal_node = GraphNode(spec, spec.goal)

    #steps = []

    #
    #
    # Code for your main method can go here.
    #
    # Your code should find a sequence of RobotConfig objects such that all configurations are collision free, the
    # distance between 2 successive configurations is less than 1 primitive step, the first configuration is the initial
    # state and the last configuration is the goal state.
    #
    #

    #
    # You may uncomment this line to launch visualiser once a solution has been found. This may be useful for debugging.
    # *** Make sure this line is commented out when you submit to Gradescope ***
    #
    # v = Visualiser(spec, steps)


if __name__ == '__main__':
    #main(["testcases/4g1_m2.txt", "output_4g1_m2.txt"])
    main(sys.argv[1:])