import copy
import time
from numpy import argmax
from laser_tank import LaserTankMap, DotDict

"""
Template file for you to implement your solution to Assignment 3. You should implement your solution by filling in the
following method stubs:
    run_value_iteration()
    run_policy_iteration()
    get_offline_value()
    get_offline_policy()
    get_mcts_policy()
    
You may add to the __init__ method if required, and can add additional helper methods and classes if you wish.

To ensure your code is handled correctly by the autograder, you should avoid using any try-except blocks in your
implementation of the above methods (as this can interfere with our time-out handling).

COMP3702 2020 Assignment 3 Support Code
"""


class Solver:

    def __init__(self, game_map):
        self.game_map = game_map
        self.params = game_map.params
        self.MOVES = [game_map.MOVE_FORWARD, game_map.TURN_LEFT, game_map.TURN_RIGHT]
        #
        # TODO
        # Write any environment preprocessing code you need here (e.g. storing teleport locations).
        #
        # You may also add any instance variables (e.g. root node of MCTS tree) here.
        #
        # The allowed time for this method is 1 second, so your Value Iteration or Policy Iteration implementation
        # should be in the methods below, not here.
        #

        self.values = None
        self.policy = None
        self.count = 0

    def run_value_iteration(self):
        """
        Build a value table and a policy table using value iteration, and store inside self.values and self.policy.
        """
        values = [[[0 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        policy = [[[-1 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        epsilon = self.params.epsilon
        gamma = self.params.gamma
        self.count = 0
        self.values = values
        self.policy = policy
        t0 = time.time()
        while True:
            delta = 0
            for x in range(1, self.game_map.x_size - 1):
                for y in range(1, self.game_map.y_size - 1):
                    for d in LaserTankMap.DIRECTIONS:
                        pre = values[x - 1][y - 1][d]
                        value = max([sum([p * s1[3] + p * gamma * values[s1[0] - 1][s1[1] - 1][s1[2]] for (s1, p) in
                                          self.transition([x, y, d], a)]) for a in self.MOVES])
                        action = argmax([sum([p * s1[3] + p * gamma * values[s1[0] - 1][s1[1] - 1][s1[2]]
                                              for (s1, p) in self.transition([x, y, d], a)]) for a in self.MOVES])
                        values[x - 1][y - 1][d] = value
                        policy[x - 1][y - 1][d] = action
                        delta = max(delta, abs(pre - values[x - 1][y - 1][d]))
            #self.values = values
            #self.policy = policy
            self.count += 1
            t = time.time() - t0
            if delta < epsilon or t >= 5:
                #print(t)
                break
        self.values = values
        self.policy = policy

    def policy_evaluation(self, policy, values, k = 2):
        """Return an updated utility mapping U from each state in the MDP to its
        utility, using an approximation (modified policy iteration)."""
        gamma = self.params.gamma
        for i in range(k):
        #while True:
            #delta = 0
            for x in range(1, self.game_map.x_size - 1):
                for y in range(1, self.game_map.y_size - 1):
                    for d in LaserTankMap.DIRECTIONS:
                        #pre = values[x - 1][y - 1][d]
                        value = sum([p * s1[3] + p * gamma * values[s1[0] - 1][s1[1] - 1][s1[2]]
                                     for (s1, p) in self.transition([x, y, d], policy[x - 1][y - 1][d])])
                        values[x - 1][y - 1][d] = value
                        #delta = max(delta, abs(pre - values[x - 1][y - 1][d]))
            #if delta < 0.001:
                #break
        return values

    def run_policy_iteration(self):
        """
        Build a value table and a policy table using policy iteration, and store inside self.values and self.policy.
        """
        values = [[[0 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        policy = [[['r' for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        gamma = self.params.gamma
        self.count = 0
        t0 = time.time()
        while self.count < 11:
            values = self.policy_evaluation(policy, values)
            unchanged = True
            for x in range(1, self.game_map.x_size - 1):
                for y in range(1, self.game_map.y_size - 1):
                    for d in range(4):
                        action = argmax([sum([p * s1[3] + p * gamma * values[s1[0] - 1][s1[1] - 1][s1[2]]
                                              for (s1, p) in self.transition([x, y, d], a)]) for a in self.MOVES])
                        if self.MOVES[action] != policy[x - 1][y - 1][d]:
                            policy[x - 1][y - 1][d] = self.MOVES[action]
                            unchanged = False
            t = time.time() - t0
            self.count += 1
            if unchanged:
                break
        #
        # TODO
        # Write your Policy Iteration implementation here.
        #
        # When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should stop
        # iterating either when max_delta < epsilon, or when the time limit is reached, whichever occurs first.
        #

        # store the computed values and policy
        self.values = values
        self.policy = policy

    def transition(self, state, action):
        next_states = []
        if state[2] == self.game_map.UP:
            if action == self.game_map.MOVE_FORWARD:
                next_x = state[0]
                next_y = state[1] - 1
                player_heading = state[2]
                next_state1 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] - 1
                next_y = state[1] - 1
                player_heading = state[2]
                next_state2 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] + 1
                next_y = state[1] - 1
                player_heading = state[2]
                next_state3 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] - 1
                next_y = state[1]
                player_heading = state[2]
                next_state4 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] + 1
                next_y = state[1]
                player_heading = state[2]
                next_state5 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0]
                next_y = state[1]
                player_heading = state[2]
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_state6 = [next_x, next_y, player_heading, cost]
                next_states = [(next_state1, self.params.t_success_prob),
                               (next_state2, (1 - self.params.t_success_prob) / 5),
                               (next_state3, (1 - self.params.t_success_prob) / 5),
                               (next_state4, (1 - self.params.t_success_prob) / 5),
                               (next_state5, (1 - self.params.t_success_prob) / 5),
                               (next_state6, (1 - self.params.t_success_prob) / 5)]
            if action == self.game_map.TURN_LEFT:
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_states = [([state[0], state[1], self.game_map.LEFT, cost], 1)]
            if action == self.game_map.TURN_RIGHT:
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_states = [([state[0], state[1], self.game_map.RIGHT, cost], 1)]
        if state[2] == self.game_map.DOWN:
            if action == self.game_map.MOVE_FORWARD:
                next_x = state[0]
                next_y = state[1] + 1
                player_heading = state[2]
                next_state1 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] + 1
                next_y = state[1] + 1
                player_heading = state[2]
                next_state2 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] - 1
                next_y = state[1] + 1
                player_heading = state[2]
                next_state3 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] - 1
                next_y = state[1]
                player_heading = state[2]
                next_state4 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] + 1
                next_y = state[1]
                player_heading = state[2]
                next_state5 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0]
                next_y = state[1]
                player_heading = state[2]
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_state6 = [next_x, next_y, player_heading, cost]
                next_states = [(next_state1, self.params.t_success_prob),
                               (next_state2, (1 - self.params.t_success_prob) / 5),
                               (next_state3, (1 - self.params.t_success_prob) / 5),
                               (next_state4, (1 - self.params.t_success_prob) / 5),
                               (next_state5, (1 - self.params.t_success_prob) / 5),
                               (next_state6, (1 - self.params.t_success_prob) / 5)]
            if action == self.game_map.TURN_LEFT:
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_states = [([state[0], state[1], self.game_map.RIGHT, cost], 1)]
            if action == self.game_map.TURN_RIGHT:
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_states = [([state[0], state[1], self.game_map.LEFT, cost], 1)]
        if state[2] == self.game_map.LEFT:
            if action == self.game_map.MOVE_FORWARD:
                next_x = state[0] - 1
                next_y = state[1]
                player_heading = state[2]
                next_state1 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] - 1
                next_y = state[1] - 1
                player_heading = state[2]
                next_state2 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] - 1
                next_y = state[1] + 1
                player_heading = state[2]
                next_state3 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0]
                next_y = state[1] - 1
                player_heading = state[2]
                next_state4 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0]
                next_y = state[1] + 1
                player_heading = state[2]
                next_state5 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0]
                next_y = state[1]
                player_heading = state[2]
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_state6 = [next_x, next_y, player_heading, cost]
                next_states = [(next_state1, self.params.t_success_prob),
                               (next_state2, (1 - self.params.t_success_prob) / 5),
                               (next_state3, (1 - self.params.t_success_prob) / 5),
                               (next_state4, (1 - self.params.t_success_prob) / 5),
                               (next_state5, (1 - self.params.t_success_prob) / 5),
                               (next_state6, (1 - self.params.t_success_prob) / 5)]
            if action == self.game_map.TURN_LEFT:
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_states = [([state[0], state[1], self.game_map.DOWN, cost], 1)]
            if action == self.game_map.TURN_RIGHT:
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_states = [([state[0], state[1], self.game_map.UP, cost], 1)]
        if state[2] == self.game_map.RIGHT:
            if action == self.game_map.MOVE_FORWARD:
                next_x = state[0] + 1
                next_y = state[1]
                player_heading = state[2]
                next_state1 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] + 1
                next_y = state[1] - 1
                player_heading = state[2]
                next_state2 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0] + 1
                next_y = state[1] + 1
                player_heading = state[2]
                next_state3 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0]
                next_y = state[1] - 1
                player_heading = state[2]
                next_state4 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0]
                next_y = state[1] + 1
                player_heading = state[2]
                next_state5 = self.generate_next_state(state[0], state[1], next_y, next_x, player_heading)
                next_x = state[0]
                next_y = state[1]
                player_heading = state[2]
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_state6 = [next_x, next_y, player_heading, cost]
                next_states = [(next_state1, self.params.t_success_prob),
                               (next_state2, (1 - self.params.t_success_prob) / 5),
                               (next_state3, (1 - self.params.t_success_prob) / 5),
                               (next_state4, (1 - self.params.t_success_prob) / 5),
                               (next_state5, (1 - self.params.t_success_prob) / 5),
                               (next_state6, (1 - self.params.t_success_prob) / 5)]
            if action == self.game_map.TURN_LEFT:
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_states = [([state[0], state[1], self.game_map.UP, cost], 1)]
            if action == self.game_map.TURN_RIGHT:
                if self.game_map.grid_data[state[1]][state[0]] == self.game_map.WATER_SYMBOL:
                    cost = self.params.game_over_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                elif self.game_map.grid_data[state[1]][state[0]] == self.game_map.FLAG_SYMBOL:
                    cost = 0
                else:
                    cost = self.params.move_cost
                next_states = [([state[0], state[1], self.game_map.DOWN, cost], 1)]
        return next_states

    def generate_next_state(self, x, y, next_y, next_x, player_heading):
        next_state1 = []
        if self.game_map.grid_data[next_y][next_x] == self.game_map.WATER_SYMBOL:
            cost = self.params.game_over_cost
            if self.game_map.grid_data[y][x] == self.game_map.WATER_SYMBOL:
                cost += self.params.game_over_cost
            next_state1 = [next_x, next_y, player_heading, cost]
        elif self.game_map.grid_data[next_y][next_x] == self.game_map.FLAG_SYMBOL:
            cost = 0
            next_state1 = [next_x, next_y, player_heading, cost]
        else:
            if next_x < 1 or next_x > self.params.x_size - 2 or next_y < 1 or next_y > self.params.y_size - 2:
                cost = self.params.collision_cost
                if next_x < 1:
                    next_x += 1
                if next_x > self.params.x_size - 2:
                    next_x -= 1
                if next_y < 1:
                    next_y += 1
                if next_y > self.params.y_size - 2:
                    next_y -= 1
                if self.game_map.grid_data[next_y][next_x] == self.game_map.OBSTACLE_SYMBOL:
                    cost += self.params.collision_cost
                next_state1 = [next_x, next_y, player_heading, cost]
            else:
                if self.game_map.grid_data[next_y][next_x] == self.game_map.OBSTACLE_SYMBOL:
                    cost = self.params.collision_cost
                    if self.game_map.grid_data[y][x] == self.game_map.OBSTACLE_SYMBOL:
                        cost += self.params.collision_cost
                    next_state1 = [x, y, player_heading, cost]
                elif self.handle_ice(next_y, next_x, player_heading) != False:
                    next_y, next_x, cost = self.handle_ice(next_y, next_x, player_heading)
                    next_state1 = [next_x, next_y, player_heading, cost]
                elif self.handle_teleport(next_y, next_x) != False:
                    next_y, next_x = self.handle_teleport(next_y, next_x)
                    cost = self.params.move_cost
                    next_state1 = [next_x, next_y, player_heading, cost]
                else:
                    cost = self.params.move_cost
                    next_state1 = [next_x, next_y, player_heading, cost]
        return next_state1

    def handle_ice(self, next_y, next_x, player_heading):
        if self.params.grid_data[next_y][next_x] == self.game_map.ICE_SYMBOL:
            # handle ice tile - slide until first non-ice tile or blocked
            if player_heading == self.game_map.UP:
                for i in range(next_y, -1, -1):
                    if self.params.grid_data[i][next_x] != self.game_map.ICE_SYMBOL:
                        if self.params.grid_data[i][next_x] == self.game_map.WATER_SYMBOL:
                            # slide into water - game over
                            return i, next_x, self.params.game_over_cost
                        elif self.game_map.cell_is_blocked(i, next_x):
                            # if blocked, stop on last ice cell
                            next_y = i + 1
                            return next_y, next_x, self.params.move_cost
                        else:
                            next_y = i
                            return i, next_x, self.params.move_cost
            elif player_heading == self.game_map.DOWN:
                for i in range(next_y, self.params.y_size):
                    if self.params.grid_data[i][next_x] != self.game_map.ICE_SYMBOL:
                        if self.params.grid_data[i][next_x] == self.game_map.WATER_SYMBOL:
                            # slide into water - game over
                            return i, next_x, self.params.game_over_cost
                        elif self.game_map.cell_is_blocked(i, next_x):
                            # if blocked, stop on last ice cell
                            next_y = i - 1
                            return next_y, next_x, self.params.move_cost
                        else:
                            next_y = i
                            return i, next_x, self.params.move_cost
            elif player_heading == self.game_map.LEFT:
                for i in range(next_x, -1, -1):
                    if self.params.grid_data[next_y][i] != self.game_map.ICE_SYMBOL:
                        if self.params.grid_data[next_y][i] == self.game_map.WATER_SYMBOL:
                            # slide into water - game over
                            return next_y, i, self.params.game_over_cost
                        elif self.game_map.cell_is_blocked(next_y, i):
                            # if blocked, stop on last ice cell
                            next_x = i + 1
                            return next_y, next_x, self.params.move_cost
                        else:
                            next_x = i
                            return next_y, i, self.params.move_cost
            else:
                for i in range(next_x, self.params.x_size):
                    if self.params.grid_data[next_y][i] != self.game_map.ICE_SYMBOL:
                        if self.params.grid_data[next_y][i] == self.game_map.WATER_SYMBOL:
                            # slide into water - game over
                            return next_y, i, self.params.game_over_cost
                        elif self.game_map.cell_is_blocked(next_y, i):
                            # if blocked, stop on last ice cell
                            next_x = i - 1
                            return next_y, next_x, self.params.move_cost
                        else:
                            next_x = i
                            return next_y, i, self.params.move_cost
        return False

    def handle_teleport(self, next_y, next_x):
        if self.params.grid_data[next_y][next_x] == self.game_map.TELEPORT_SYMBOL:
            # handle teleport - find the other teleporter
            tpy, tpx = (None, None)
            for i in range(self.params.y_size):
                for j in range(self.params.x_size):
                    if self.params.grid_data[i][j] == self.game_map.TELEPORT_SYMBOL and i != next_y and j != next_x:
                        tpy, tpx = (i, j)
                        break
                if tpy is not None:
                    break
            if tpy is None:
                raise Exception("LaserTank Map Error: Unmatched teleport symbol")
            next_y, next_x = (tpy, tpx)
            return next_y, next_x
        return False

    def get_offline_value(self, state):
        """
        Get the value of this state.
        :param state: a LaserTankMap instance
        :return: V(s) [a floating point number]
        """
        player_x = state.player_x
        player_y = state.player_y
        player_heading = state.player_heading
        return self.values[player_x - 1][player_y - 1][state.player_heading]
        #
        # TODO
        # Write code to return the value of this state based on the stored self.values
        #
        # You can assume that either run_value_iteration( ) or run_policy_iteration( ) has been called before this
        # method is called.
        #
        # When this method is called, you are allowed up to 1 second of compute time.
        #

        pass

    def get_offline_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """
        player_x = state.player_x
        player_y = state.player_y
        player_heading = state.player_heading
        if self.policy[player_x - 1][player_y - 1][player_heading] == 0 or \
                self.policy[player_x - 1][player_y - 1][player_heading] == 1 or \
                self.policy[player_x - 1][player_y - 1][player_heading] == 2:
            return LaserTankMap.MOVES[self.policy[player_x - 1][player_y - 1][player_heading]]
        else:
            return self.policy[player_x - 1][player_y - 1][player_heading]
        #
        # TODO
        # Write code to return the optimal action to be performed at this state based on the stored self.policy
        #
        # You can assume that either run_value_iteration( ) or run_policy_iteration( ) has been called before this
        # method is called.
        #
        # When this method is called, you are allowed up to 1 second of compute time.
        #

        pass

    def get_mcts_policy(self, state):
        """
        Choose an action to be performed using online MCTS.
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """

        #
        # TODO
        # Write your Monte-Carlo Tree Search implementation here.
        #
        # Each time this method is called, you are allowed up to [state.time_limit] seconds of compute time - make sure
        # you stop searching before this time limit is reached.
        #

        pass


