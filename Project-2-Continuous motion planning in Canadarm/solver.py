import copy
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

    """
    Function design of generate_random_config and collision_free, steps_generator and the search part of prm were 
    reference from github open code source, it is only a reference to some basic methods and ideas. 
    link: https://github.com/nanyangeva/COMP7702_AI.
    The key parts of the code implementation, such as sampling from the workspace, PRM, find path, and resolving
    multiple grappled points issues, were all done independently by myself.
    """
    def generate_random_config(self, config):
        if config.ee1_grappled:
            ee1x, ee1y = config.get_ee1()
        else:
            ee2x, ee2y = config.get_ee2()
        config_lengths = []
        for i in range(self.spec.num_segments):
            length = np.random.uniform(self.spec.min_lengths[i], self.spec.max_lengths[i])
            config_lengths.append(length)
        config_angles = []
        random_degree = np.random.randint(-165, 165, self.spec.num_segments)
        for i in range(self.spec.num_segments):
            angle = Angle(degrees=random_degree[i])
            config_angles.append(angle)
        if config.ee1_grappled:
            new_config = make_robot_config_from_ee1(ee1x, ee1y, config_angles, config_lengths, ee1_grappled=True)
        else:
            new_config = make_robot_config_from_ee2(ee2x, ee2y, config_angles, config_lengths, ee2_grappled=True)
        while not (self.valid_config(new_config)):
            new_config = self.generate_random_config(config)
        return new_config

    def generate_random_configs(self, based_config, n):
        bunch_config = {}
        i = 0
        while i < n:
            config = self.generate_random_config(based_config)
            bunch_config[config] = config
            i = i + 1
        return bunch_config

    def generate_random_config2(self, config):
        if config.ee1_grappled:
            ee1x, ee1y = config.get_ee1()
        else:
            ee2x, ee2y = config.get_ee2()
        config_lengths = []
        for i in range(self.spec.num_segments):
            length = np.random.uniform(self.spec.min_lengths[i], self.spec.max_lengths[i])
            config_lengths.append(length)
        config_angles = []
        random_degree = np.random.randint(-165, 165, self.spec.num_segments)
        random_degree2 = np.random.randint(1, 179, 1)
        angle = Angle(degrees=random_degree2[0])
        config_angles.append(angle)
        for i in range(self.spec.num_segments - 1):
            angle = Angle(degrees=random_degree[i])
            config_angles.append(angle)
        if config.ee1_grappled:
            new_config = make_robot_config_from_ee1(ee1x, ee1y, config_angles, config_lengths, ee1_grappled=True)
        else:
            new_config = make_robot_config_from_ee2(ee2x, ee2y, config_angles, config_lengths, ee2_grappled=True)
        while not (self.valid_config(new_config)):
            new_config = self.generate_random_config(config)
        return new_config

    def generate_random_configs2(self, based_config, n):
        bunch_config = {}
        i = 0
        while i < n:
            config = self.generate_random_config2(based_config)
            bunch_config[config] = config
            i = i + 1
        return bunch_config

    def bfs(self, graph, initial, goal):
        # the set to store explored config
        explored = set()
        # the list to store fringe config
        fringe_list = [[initial]]

        while len(fringe_list) > 0:
            # pop first node in queue
            points = fringe_list.pop(0)
            fringe_point = points[-1]
            if hash(tuple(fringe_point)) not in explored:
                # add node to explored list
                explored.add(hash(tuple(fringe_point)))
                # if the current node has children
                if fringe_point in graph.keys():
                    for child in graph[fringe_point]:
                        solution = copy.deepcopy(points)
                        solution.append(child)
                        if child[0] == goal[0] and child[1] == goal[1]:
                            solution_found = True
                            return solution, solution_found
                        else:
                            fringe_list.append(solution)
        solution_found = False
        return [], solution_found

    def workspace_sample(self, config, goal_config, n):
        last_points = []
        graph = {}
        bunch_config = {}
        i = 0
        while i < n:
            random_config = self.generate_random_config(config)
            if config.ee1_grappled:
                last_point = random_config.points[-1]
            else:
                last_point = random_config.points[0]
            last_points.append(last_point)
            i = i + 1
        if config.ee1_grappled:
            goal_point = goal_config.points[-1]
            initial_point = config.points[-1]
        else:
            goal_point = goal_config.points[0]
            initial_point = config.points[0]
        last_points.append(initial_point)
        last_points.append(goal_point)
        for point in last_points:
            if point not in graph:
                graph[point] = []
            for point_another in last_points:
                distance = np.sqrt(np.square(point[0] - point_another[0]) + np.square(point[1] - point_another[1]))
                if distance == 0 or (point[0] == point_another[0] and point[1] == point_another[1]):
                    continue
                if 0 < distance <= 0.2:
                    graph[point].append(point_another)
        path = self.bfs(graph, initial_point, goal_point)
        for point in path[0]:
            i = 0
            while i < 30:
                random_config = self.generate_random_config(config)
                if config.ee1_grappled:
                    last_point = random_config.points[-1]
                else:
                    last_point = random_config.points[0]
                distance = np.sqrt(np.square(point[0] - last_point[0]) + np.square(point[1] - last_point[1]))
                if 0 < distance <= 0.1:
                    bunch_config[random_config] = random_config
                    i = i + 1
        return bunch_config

    def valid_config(self, config):
        return (test_angle_constraints(config, self.spec)
                and test_obstacle_collision(config, self.spec, self.spec.obstacles)
                and test_self_collision(config, self.spec)
                and test_environment_bounds(config)
                and test_length_constraints(config, self.spec))

    def generate_random_config_near_goal(self, goal_config, n):
        ee1x, ee1y = goal_config.get_ee1()
        ee2x, ee2y = goal_config.get_ee2()
        # generate ee1_length
        bunch_config = {}
        #random_length = 0
        j = 0
        while j < n:
            config_lengths = []
            for i in range(self.spec.num_segments):
                if goal_config.lengths[i] == self.spec.max_lengths[i]:
                    random_length = goal_config.lengths[i] - 0.001 * j
                elif goal_config.lengths[i] == self.spec.min_lengths[i]:
                    random_length = goal_config.lengths[i] + 0.001 * j
                else:
                    if j < n / 2:
                        random_length = goal_config.lengths[i] - 0.001 * j
                    else:
                        random_length = goal_config.lengths[i] + 0.001*(j + 1 - n / 2)
                config_lengths.append(random_length)
            # generate ee1_angles
            config_angles = []
            for i in range(self.spec.num_segments):
                if j < n / 2:
                    if goal_config.ee1_grappled:
                        random_angle = goal_config.ee1_angles[i] - 0.02 * j
                    else:
                        random_angle = goal_config.ee2_angles[i] - 0.02 * j
                    config_angles.append(random_angle)
                else:
                    if goal_config.ee1_grappled:
                        random_angle = goal_config.ee1_angles[i] + 0.02 * (j + 1 - n / 2)
                    else:
                        random_angle = goal_config.ee2_angles[i] + 0.02 * (j + 1 - n / 2)
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

    def generate_bridge_config(self, config, grappled_point):
        if config.ee1_grappled:
            ee1x, ee1y = config.get_ee1()
        else:
            ee2x, ee2y = config.get_ee2()
        config_lengths = []
        for i in range(self.spec.num_segments - 1):
            length = np.random.uniform(self.spec.min_lengths[i], self.spec.max_lengths[i])
            config_lengths.append(length)
        config_angles = []
        random_degree = np.random.randint(-165, 165, self.spec.num_segments)
        for i in range(self.spec.num_segments - 1):
            angle = Angle(degrees=random_degree[i])
            config_angles.append(angle)
        if config.ee1_grappled:
            new_config = make_robot_config_from_ee1(ee1x, ee1y, config_angles, config_lengths, ee1_grappled=True)
        else:
            new_config = make_robot_config_from_ee2(ee2x, ee2y, config_angles, config_lengths, ee2_grappled=True)
        if config.ee1_grappled:
            vector1 = [grappled_point[0] - new_config.points[-1][0], grappled_point[1] - new_config.points[-1][1]]
            vector2 = [new_config.points[-1][0] - new_config.points[-2][0],
                       new_config.points[-1][1] - new_config.points[-2][1]]
            angle_last = Angle.acos((vector1[0] * vector2[0] + vector1[1] * vector2[1]) /
                                    (np.sqrt(np.square(vector1[0]) + np.square(vector1[1])) *
                                     np.sqrt(np.square(vector2[0]) + np.square(vector2[1]))))
            length_last = np.sqrt(np.square(new_config.points[-1][0] - grappled_point[0]) +
                                  np.square(new_config.points[-1][1] - grappled_point[1]))
            net_angle = angle_last
            for i in range(len(new_config.ee1_angles)):
                net_angle = net_angle + new_config.ee1_angles[i]
            x_new = new_config.points[-1][0] + (length_last * math.cos(net_angle.in_radians()))
        else:
            vector1 = [grappled_point[0] - new_config.points[0][0], grappled_point[1] - new_config.points[0][1]]
            vector2 = [new_config.points[0][0] - new_config.points[1][0],
                       new_config.points[0][1] - new_config.points[1][1]]
            angle_last = Angle.acos((vector1[0] * vector2[0] + vector1[1] * vector2[1]) /
                                    (np.sqrt(np.square(vector1[0]) + np.square(vector1[1])) *
                                     np.sqrt(np.square(vector2[0]) + np.square(vector2[1]))))
            length_last = np.sqrt(np.square(new_config.points[0][0] - grappled_point[0]) +
                                  np.square(new_config.points[0][1] - grappled_point[1]))
            net_angle = angle_last
            for i in range(len(new_config.ee2_angles)):
                net_angle = net_angle + new_config.ee2_angles[i]
            x_new = new_config.points[0][0] + (length_last * math.cos(net_angle.in_radians()))
        if grappled_point[0] - 0.0001 <= x_new <= grappled_point[0] + 0.0001:
            angle_last = Angle(degrees=angle_last.in_degrees())
        else:
            angle_last = Angle(degrees=-angle_last.in_degrees())
        config_angles.append(angle_last)
        if config.ee1_grappled:
            config_lengths.append(length_last)
            bridge_config_ee1 = make_robot_config_from_ee1(ee1x, ee1y, config_angles, config_lengths, ee1_grappled=True)
            ee2_config_angles = bridge_config_ee1.ee2_angles
            bridge_config_ee2 = make_robot_config_from_ee2(grappled_point[0], grappled_point[1],
                                                           ee2_config_angles, config_lengths, ee2_grappled=True)
        else:
            config_lengths.insert(0, length_last)
            bridge_config_ee1 = make_robot_config_from_ee2(ee2x, ee2y, config_angles, config_lengths, ee2_grappled=True)
            ee1_config_angles = bridge_config_ee1.ee1_angles
            bridge_config_ee2 = make_robot_config_from_ee1(grappled_point[0], grappled_point[1],
                                                           ee1_config_angles, config_lengths, ee1_grappled=True)
        return [bridge_config_ee1, bridge_config_ee2]

    def generate_bridge_ee_config(self, config, grappled_point):
        bridge_config = self.generate_bridge_config(config, grappled_point)
        while not (self.valid_config(bridge_config[0]) and self.spec.min_lengths[0] <= bridge_config[0].lengths[-1]
                   <= self.spec.max_lengths[0]):
            bridge_config = self.generate_bridge_config(config, grappled_point)
        return bridge_config

    def generate_vector(self, config):
        config_vector = []
        if config.ee1_grappled:
            config_vector.extend(config.ee1_angles)
            config_vector.extend(config.lengths)
        else:
            config_vector.extend(config.ee2_angles)
            config_vector.extend(config.lengths)
        return config_vector

    def calculate_config_distance(self, config1: RobotConfig, config2: RobotConfig):
        points1 = config1.points
        points2 = config2.points
        distance = 0
        for i in range(self.spec.num_segments + 1):
            distance += np.sqrt(np.square(points1[i][0] - points2[i][0]) + np.square(points1[i][1] - points2[i][1]))
        return distance

    def collision_free(self, config1, config2):
        config_new = copy.deepcopy(config1)
        diff = np.array(self.generate_vector(config_new)) - np.array(
            self.generate_vector(config2))
        angles_diff = diff[0:self.spec.num_segments]
        angle_min_step = Angle(radians=0.02)
        steps = [item.in_radians() / angle_min_step.in_radians() for item in angles_diff]

        length_min_step = 0.02
        length_steps = diff[self.spec.num_segments:] / length_min_step
        steps.extend(length_steps)

        while not (test_config_equality(config_new, config2, self.spec)):
            if config_new.ee1_grappled:
                for i in range(self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.ee1_angles[i] += np.sign(steps[i]) * (-1) * angle_min_step
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.ee1_angles[i] = config2.ee1_angles[i]
                        steps[i] = 0
                for i in range(self.spec.num_segments, 2 * self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.lengths[i - self.spec.num_segments] += np.sign(steps[i]) * (-1) * length_min_step
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.lengths[i - self.spec.num_segments] = config2.lengths[i - self.spec.num_segments]
                        steps[i] = 0

                config_new = RobotConfig(config_new.lengths, ee1x=config_new.get_ee1()[0], ee1y=config_new.get_ee1()[1],
                                         ee1_angles=config_new.ee1_angles,
                                         ee1_grappled=True)

            else:
                for i in range(self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.ee2_angles[i] += np.sign(steps[i]) * (-1) * angle_min_step
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.ee2_angles[i] = config2.ee2_angles[i]
                        steps[i] = 0
                for i in range(self.spec.num_segments, 2 * self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.lengths[i - self.spec.num_segments] += np.sign(steps[i]) * (-1) * length_min_step
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.lengths[i - self.spec.num_segments] = config2.lengths[i - self.spec.num_segments]
                        steps[i] = 0

                config_new = RobotConfig(config_new.lengths, ee2x=config_new.get_ee2()[0], ee2y=config_new.get_ee2()[1],
                                         ee2_angles=config_new.ee2_angles,
                                         ee2_grappled=True)

            if not (self.valid_config(config_new)):
                return False
        return True

    def generate_mid_config(self, p: RobotConfig, q: RobotConfig):
        p_vector = self.generate_vector(p)
        q_vector = self.generate_vector(q)
        q_new_vector = []
        for i in range(len(p_vector)):
            if isinstance(p_vector[i], Angle):
                radians_val = (q_vector[i].in_radians() + p_vector[i].in_radians()) / 2
                angle = Angle(radians=radians_val)
                q_new_vector.append(angle)
            else:
                q_new_vector.append((q_vector[i] + p_vector[i]) / 2)

        config_angles = []
        for i in range(len(p.ee1_angles)):
            config_angles.append(q_new_vector[i])

        config_lengths = []
        for i in range(len(p.ee1_angles), len(p.ee1_angles) + len(p.lengths)):
            config_lengths.append(q_new_vector[i])

        if q.ee1_grappled:
            random_config = make_robot_config_from_ee1(q.points[0][0], q.points[0][1], config_angles, config_lengths,
                                                       ee1_grappled=True)
        else:
            random_config = make_robot_config_from_ee2(q.points[-1][0], q.points[-1][1], config_angles, config_lengths,
                                                       ee2_grappled=True)

        return random_config

    def prm(self, initial_config, goal_config, n, no_collision_distance, flag):
        bunch_configs = {}
        if flag == 1:
            bunch_configs_random = self.generate_random_configs(initial_config, n)
            bunch_configs.update(bunch_configs_random)
            bunch_configs[initial_config] = initial_config
            bunch_configs[goal_config] = goal_config
            bunch_configs_near_goal = self.generate_random_config_near_goal(goal_config, n / 2)
            bunch_configs.update(bunch_configs_near_goal)
        if flag == 2:
            bunch_configs_workspace = self.workspace_sample(initial_config, goal_config, n)
            bunch_configs.update(bunch_configs_workspace)
            bunch_configs[initial_config] = initial_config
            bunch_configs[goal_config] = goal_config
            bunch_configs_near_goal = self.generate_random_config_near_goal(goal_config, n / 4)
            bunch_configs.update(bunch_configs_near_goal)
        if flag == 3:
            bunch_configs_random = self.generate_random_configs2(initial_config, n)
            bunch_configs.update(bunch_configs_random)
            bunch_configs[initial_config] = initial_config
            bunch_configs[goal_config] = goal_config
            bunch_configs_near_goal = self.generate_random_config_near_goal(goal_config, n / 2)
            bunch_configs.update(bunch_configs_near_goal)
        distance_list = []
        for config in bunch_configs:
                for config_another in bunch_configs:
                    if test_config_equality(config, config_another, self.spec):
                        distance = 0
                    else:
                        distance = self.calculate_config_distance(config, config_another)
                        distance_list.append(distance)
        distance_list.sort(reverse=False)
        threshold_distance = distance_list[int(0.2 * len(distance_list))]
        print(threshold_distance)

        graph = {}
        for config in bunch_configs:
            if config not in graph:
                graph[config] = []
            for config_another, v in bunch_configs.items():
                distance = self.calculate_config_distance(config, config_another)
                if distance == 0 or test_config_equality(config, config_another, self.spec):
                    continue
                if 0 < distance <= no_collision_distance:
                    if self.collision_free(config, config_another):
                        graph[config].append(config_another)
                    else:
                        mid_config = self.generate_mid_config(config, config_another)
                        distance = self.calculate_config_distance(config, mid_config)
                        if distance <= 0.6 and self.collision_free(config, mid_config):
                            graph[config].append(mid_config)

        return graph

    def steps_generator(self, config1, config2):
        result_list = []
        config_new = copy.deepcopy(config1)
        diff = np.array(self.generate_vector(config_new)) - np.array(
            self.generate_vector(config2))
        angles_diff = diff[0:self.spec.num_segments]
        angle_min_step = Angle(radians=0.001)
        steps = [item.in_radians() / angle_min_step.in_radians() for item in angles_diff]
        length_min_step = 0.001
        length_steps = diff[self.spec.num_segments:] / length_min_step
        steps.extend(length_steps)

        config_first = copy.deepcopy(config1)
        result_list.append(config_first)
        while not (test_config_equality(config_new, config2, self.spec)):
            if config_new.ee1_grappled:
                for i in range(self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.ee1_angles[i] += np.sign(steps[i]) * (-1) * angle_min_step
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.ee1_angles[i] = config2.ee1_angles[i]
                        steps[i] = 0
                for i in range(self.spec.num_segments, 2 * self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.lengths[i - self.spec.num_segments] += np.sign(steps[i]) * (-1) * length_min_step
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.lengths[i - self.spec.num_segments] = config2.lengths[i - self.spec.num_segments]
                        steps[i] = 0
                config_new = RobotConfig(config_new.lengths, ee1x=config_new.get_ee1()[0], ee1y=config_new.get_ee1()[1],
                                         ee1_angles=config_new.ee1_angles,
                                         ee1_grappled=True)
            else:
                for i in range(self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.ee2_angles[i] += np.sign(steps[i]) * (-1) * angle_min_step
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.ee2_angles[i] = config2.ee2_angles[i]
                        steps[i] = 0
                for i in range(self.spec.num_segments, 2 * self.spec.num_segments):
                    if np.abs(steps[i]) > 1:
                        config_new.lengths[i - self.spec.num_segments] += np.sign(steps[i]) * (-1) * length_min_step
                        steps[i] += np.sign(steps[i]) * (-1)
                    else:
                        config_new.lengths[i - self.spec.num_segments] = config2.lengths[i - self.spec.num_segments]
                        steps[i] = 0
                config_new = RobotConfig(config_new.lengths, ee2x=config_new.get_ee2()[0], ee2y=config_new.get_ee2()[1],
                                         ee2_angles=config_new.ee2_angles,
                                         ee2_grappled=True)

            config_record = copy.deepcopy(config_new)
            result_list.append(config_record)
        return result_list

    def steps(self, config_list):
        steps = []
        for i in range(len(config_list) - 1):
            result_list = self.steps_generator(config_list[i], config_list[i + 1])
            steps.extend(result_list)
        return steps

    def find_graph_path(self, graph, initial, goal_config):
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
    solution_found1 = False
    solution_found2 = False
    solution_found3 = False
    solution = []
    solution1 = []
    solution2 = []
    solution3 = []
    solution4 = []
    i = 0

    start = time.process_time()
    if spec.num_grapple_points < 2:
        while not solution_found and i < 3:
            print("PRM round", i + 1, "start")
            graph = solver.prm(initial_config, goal_config, 200, 0.5, 1)
            print("Path find round", i + 1, "start")
            solution, solution_found = solver.find_graph_path(graph, initial_config, goal_config)
            i += 1
    elif spec.num_grapple_points == 2:
        grappled_point = spec.grapple_points[1]
        inter_goal = solver.generate_bridge_ee_config(initial_config, grappled_point)
        print(inter_goal[1])
        #for j in [0, 1]:
        while not solution_found and i < 3:
            print(0, "PRM round", i + 1, "start")
            graph = solver.prm(initial_config, inter_goal[0], 100, 0.4, 1)
            print(0, "Path find round", i + 1, "start")
            solution1, solution_found = solver.find_graph_path(graph, initial_config, inter_goal[0])
            i += 1
        i = 0
        while not solution_found1 and i < 3:
            print(1, "PRM round", i + 1, "start")
            graph = solver.prm(inter_goal[1], goal_config, 100, 0.4, 1)
            print(1, "Path find round", i + 1, "start")
            solution2, solution_found1 = solver.find_graph_path(graph, inter_goal[1], goal_config)
            i += 1
    elif spec.num_grapple_points == 3 and spec.num_segments == 4:
        grappled_point1 = spec.grapple_points[1]
        grappled_point2 = spec.grapple_points[2]
        inter_goal1 = solver.generate_bridge_ee_config(initial_config, grappled_point1)
        inter_goal2 = solver.generate_bridge_ee_config(inter_goal1[1], grappled_point2)
        print(inter_goal1[1])
        #for j in [0, 1]:
        while not solution_found and i < 3:
            print(0, "PRM round", i + 1, "start")
            graph = solver.prm(initial_config, inter_goal1[0], 150, 0.5, 1)
            print(0, "Path find round", i + 1, "start")
            solution1, solution_found = solver.find_graph_path(graph, initial_config, inter_goal1[0])
            i += 1
        i = 0
        while not solution_found1 and i < 3:
            print(1, "PRM round", i + 1, "start")
            graph = solver.prm(inter_goal1[1], inter_goal2[0], 250, 0.5, 1)
            print(1, "Path find round", i + 1, "start")
            solution2, solution_found1 = solver.find_graph_path(graph, inter_goal1[1], inter_goal2[0])
            i += 1
        i = 0
        while not solution_found2 and i < 3:
            print(2, "PRM round", i + 1, "start")
            graph = solver.prm(inter_goal2[1], goal_config, 150, 0.5, 1)
            print(2, "Path find round", i + 1, "start")
            solution3, solution_found2 = solver.find_graph_path(graph, inter_goal2[1], goal_config)
            i += 1
    elif spec.num_grapple_points == 3 and spec.num_segments == 5:
        grappled_point1 = spec.grapple_points[1]
        grappled_point2 = spec.grapple_points[2]
        inter_goal1 = solver.generate_bridge_ee_config(initial_config, grappled_point1)
        inter_goal2 = solver.generate_bridge_ee_config(inter_goal1[1], grappled_point2)
        print(inter_goal1[0])
        #for j in [0, 1]:
        while not solution_found and i < 3:
            print(0, "PRM round", i + 1, "start")
            graph = solver.prm(initial_config, inter_goal1[0], 220, 0.5, 3)
            print(0, "Path find round", i + 1, "start")
            solution1, solution_found = solver.find_graph_path(graph, initial_config, inter_goal1[0])
            i += 1
        i = 0
        while not solution_found1 and i < 3:
            print(1, "PRM round", i + 1, "start")
            graph = solver.prm(inter_goal1[1], inter_goal2[0], 350, 0.5, 3)
            print(1, "Path find round", i + 1, "start")
            solution2, solution_found1 = solver.find_graph_path(graph, inter_goal1[1], inter_goal2[0])
            i += 1
        i = 0
        while not solution_found2 and i < 3:
            print(2, "PRM round", i + 1, "start")
            graph = solver.prm(inter_goal2[1], goal_config, 220, 0.5, 3)
            print(2, "Path find round", i + 1, "start")
            solution3, solution_found2 = solver.find_graph_path(graph, inter_goal2[1], goal_config)
            i += 1
    elif spec.num_grapple_points == 4:
        grappled_point1 = spec.grapple_points[1]
        grappled_point2 = spec.grapple_points[2]
        grappled_point3 = spec.grapple_points[3]
        inter_goal1 = solver.generate_bridge_ee_config(initial_config, grappled_point1)
        inter_goal2 = solver.generate_bridge_ee_config(inter_goal1[1], grappled_point2)
        inter_goal3 = solver.generate_bridge_ee_config(inter_goal2[1], grappled_point3)
        print(inter_goal1[1])
        #for j in [0, 1]:
        while not solution_found and i < 3:
            print(0, "PRM round", i + 1, "start")
            graph = solver.prm(initial_config, inter_goal1[0], 150, 0.5, 1)
            print(0, "Path find round", i + 1, "start")
            solution1, solution_found = solver.find_graph_path(graph, initial_config, inter_goal1[0])
            i += 1
        i = 0
        while not solution_found1 and i < 3:
            print(1, "PRM round", i + 1, "start")
            graph = solver.prm(inter_goal1[1], inter_goal2[0], 250, 0.5, 1)
            print(1, "Path find round", i + 1, "start")
            solution2, solution_found1 = solver.find_graph_path(graph, inter_goal1[1], inter_goal2[0])
            i += 1
        i = 0
        while not solution_found2 and i < 3:
            print(2, "PRM round", i + 1, "start")
            graph = solver.prm(inter_goal2[1], inter_goal3[0], 100, 0.6, 1)
            print(2, "Path find round", i + 1, "start")
            solution3, solution_found2 = solver.find_graph_path(graph, inter_goal2[1], inter_goal3[0])
            i += 1
        i = 0
        while not solution_found3 and i < 3:
            print(3, "PRM round", i + 1, "start")
            graph = solver.prm(inter_goal3[1], goal_config, 150, 0.5, 1)
            print(3, "Path find round", i + 1, "start")
            solution4, solution_found3 = solver.find_graph_path(graph, inter_goal3[1], goal_config)
            i += 1

    # print("Solution found at", datetime.datetime.now())
    # found eventually for every leg
    if solution_found or i == 3:
        if spec.num_grapple_points < 2:
            steps = solver.steps(solution)
            write_robot_config_list_to_file(output_file, steps)
        elif spec.num_grapple_points == 2:
            steps1 = solver.steps(solution1)
            steps2 = solver.steps(solution2)
            steps1.extend(steps2)
            #solution2.extend(steps)
            write_robot_config_list_to_file(output_file, steps1)
        elif spec.num_grapple_points == 3:
            steps1 = solver.steps(solution1)
            steps2 = solver.steps(solution2)
            steps3 = solver.steps(solution3)
            steps1.extend(steps2)
            steps1.extend(steps3)
            #solution2.extend(steps)
            write_robot_config_list_to_file(output_file, steps1)
        elif spec.num_grapple_points == 4:
            steps1 = solver.steps(solution1)
            steps2 = solver.steps(solution2)
            steps3 = solver.steps(solution3)
            steps4 = solver.steps(solution4)
            steps1.extend(steps2)
            steps1.extend(steps3)
            steps1.extend(steps4)
            #solution2.extend(steps)
            write_robot_config_list_to_file(output_file, steps1)
    else:
        print("Solution not found")
    elapsed = (time.process_time() - start)
    print("Time used:", elapsed)


if __name__ == '__main__':
    main(["testcases/5g3_m2.txt", "output_5g3_m20.txt"])
    #main(sys.argv[1:])
