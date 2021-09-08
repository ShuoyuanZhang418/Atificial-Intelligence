import copy
import numpy as np
import random
import time

# Directions
from laser_tank import LaserTankMap

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

OBSTACLES = [(1, 1)]
EXIT_STATE = (-1, -1)

class Grid:

    def __init__(self):
        self.x_size = 4
        self.y_size = 3
        self.p = 0.8
        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.rewards = {(3, 1): -100, (3, 2): 1}
        self.discount = 0.9

        self.states = list((x, y) for x in range(self.x_size) for y in range(self.y_size))
        self.states.append(EXIT_STATE)
        for obstacle in OBSTACLES:
            self.states.remove(obstacle)

    def attempt_move(self, s, a):
        # Check absorbing state
        if s == EXIT_STATE:
            return s

        # Default: no movement
        result = s 

        # Check borders
        """
        Write code here to check if applying an action 
        keeps the agent with the boundary
        """

        # Check obstacle cells
        """
        Write code here to check if applying an action 
        moves the agent into an obstacle cell
        """

        return result

    def stoch_action(self, a):
        # Stochasitc actions probability distributions
        if a == RIGHT: 
            stoch_a = {RIGHT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        if a == UP:
            stoch_a = {UP: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        if a == LEFT:
            stoch_a = {LEFT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        if a == DOWN:
            stoch_a = {DOWN: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        return stoch_a

    def get_reward(self, s):
        if s == EXIT_STATE:
            return 0

        if s in self.rewards:
            return self.rewards[s]
        else:
            return 0

class ValueIteration:
    def __init__(self, grid):
        self.grid = Grid()
        self.values = {state: 0 for state in self.grid.states}

    def next_iteration(self):
        new_values = dict()
        """
        Write code here to imlpement the VI value update
        Iterate over self.grid.states and self.grid.actions
        Use stoch_action(a) and attempt_move(s,a)
        """
        self.values = new_values

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)


class PolicyIteration:
    def __init__(self, grid):
        self.grid = Grid()
        self.values = {state: 0 for state in self.grid.states}
        self.policy = {pi: RIGHT for pi in self.grid.states}
        self.r = [0 for s in self.grid.states]
        idx = 0
        for state in self.grid.states: 
            if state in self.grid.rewards.keys(): 
                self.r[idx] = self.grid.rewards[state]
            idx = idx+1
        print('r is ', self.r)
        
    def policy_evaluation(self):
        P_pi = np.zeros((len(self.grid.states),len(self.grid.states)), dtype=np.float64)
        """
        Write code to generate the state transition probability matrix 
        under a policy pi here; this is matrix P_pi initialised above
        Ensure the Pi_p index order is consistent with r 
        """ 
        
        A = np.identity(len(self.grid.states)) - self.grid.discount*P_pi
        v = np.linalg.solve(A, self.r)
        self.values = {state: v for state in self.grid.states}

    def policy_improvement(self):
        """
        Write code to extract the best policy for a given value function here
        """ 
        return

    def convergence_check(self):
        """
        Write code to check if PI has converged here
        """   
        return
        
    def print_values(self):
        for state, value in self.values.items():
            print(state, value)
    
    def print_policy(self):
        for state, policy in self.policy.items():
            print(state, policy)

    def transition_vi_pi(self, game_map, move):

        action_reward = 0
        action_value = 0
        next_wrong_ys = []
        next_wrong_xs = []
        if move == LaserTankMap.MOVE_FORWARD:

            # get coordinates for next cell
            if game_map.player_heading == LaserTankMap.UP:
                next_y = game_map.player_y - 1
                next_x = game_map.player_x

                if next_y < 1:
                    action_reward += game_map.collision_cost * game_map.t_success_prob
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_success_prob
                else:
                    this_action_reward, this_action_value = self.get_reward_value(
                        game_map, next_y, next_x, game_map.player_heading)
                    action_reward += this_action_reward * game_map.t_success_prob
                    action_value += this_action_value * game_map.t_success_prob

                if game_map.player_y - 1 < 1 or game_map.player_x - 1 < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y - 1)
                    next_wrong_xs.append(game_map.player_x - 1)

                if game_map.player_y - 1 < 1 or game_map.player_x + 1 >= game_map.x_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y - 1)
                    next_wrong_xs.append(game_map.player_x + 1)

                if game_map.player_y < 1 or game_map.player_x - 1 < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y)
                    next_wrong_xs.append(game_map.player_x - 1)

                if game_map.player_y < 1 or game_map.player_x + 1 >= game_map.x_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y)
                    next_wrong_xs.append(game_map.player_x + 1)

                if game_map.player_y < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y)
                    next_wrong_xs.append(game_map.player_x)

            elif game_map.player_heading == LaserTankMap.DOWN:

                next_y = game_map.player_y + 1
                next_x = game_map.player_x

                if next_y >= game_map.y_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_success_prob
                    action_value += \
                        self.values[game_map.player_x - 1][
                            game_map.player_y - 1][
                            game_map.player_heading] * game_map.t_success_prob
                else:
                    this_action_reward, this_action_value = self.get_reward_value(
                        game_map, next_y, next_x, game_map.player_heading)
                    action_reward += this_action_reward * game_map.t_success_prob
                    action_value += this_action_value * game_map.t_success_prob

                if game_map.player_y + 1 >= game_map.y_size - 1 or game_map.player_x - 1 < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y + 1)
                    next_wrong_xs.append(game_map.player_x - 1)

                if game_map.player_y + 1 >= game_map.y_size - 1 or game_map.player_x + 1 >= game_map.x_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y + 1)
                    next_wrong_xs.append(game_map.player_x + 1)

                if game_map.player_y >= game_map.y_size - 1 or game_map.player_x - 1 < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                                1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y)
                    next_wrong_xs.append(game_map.player_x - 1)

                if game_map.player_y >= game_map.y_size - 1 or game_map.player_x + 1 >= game_map.x_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                                1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y)
                    next_wrong_xs.append(game_map.player_x + 1)

                if game_map.player_y >= game_map.y_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y)
                    next_wrong_xs.append(game_map.player_x)

            elif game_map.player_heading == LaserTankMap.LEFT:
                next_y = game_map.player_y
                next_x = game_map.player_x - 1

                if next_x < 1:
                    action_reward += game_map.collision_cost * game_map.t_success_prob
                    action_value += \
                        self.values[game_map.player_x - 1][
                            game_map.player_y - 1][
                            game_map.player_heading] * game_map.t_success_prob
                else:
                    this_action_reward, this_action_value = self.get_reward_value(
                        game_map, next_y, next_x, game_map.player_heading)
                    action_reward += this_action_reward * game_map.t_success_prob
                    action_value += this_action_value * game_map.t_success_prob

                if game_map.player_x - 1 < 1 or game_map.player_y - 1 < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y - 1)
                    next_wrong_xs.append(game_map.player_x - 1)

                if game_map.player_x - 1 < 1 or game_map.player_y + 1 >= game_map.y_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y + 1)
                    next_wrong_xs.append(game_map.player_x - 1)

                if game_map.player_x < 1 or game_map.player_y - 1 < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y - 1)
                    next_wrong_xs.append(game_map.player_x)

                if game_map.player_x < 1 or game_map.player_y + 1 >= game_map.y_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y + 1)
                    next_wrong_xs.append(game_map.player_x)

                if game_map.player_x < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y)
                    next_wrong_xs.append(game_map.player_x)

            else:
                next_y = game_map.player_y
                next_x = game_map.player_x + 1

                if next_x >= game_map.x_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_success_prob
                    action_value += \
                        self.values[game_map.player_x - 1][
                            game_map.player_y - 1][
                            game_map.player_heading] * game_map.t_success_prob
                else:
                    this_action_reward, this_action_value = self.get_reward_value(
                        game_map, next_y, next_x, game_map.player_heading)
                    action_reward += this_action_reward * game_map.t_success_prob
                    action_value += this_action_value * game_map.t_success_prob

                if next_x + 1 >= game_map.x_size - 1 or game_map.player_y - 1 < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y - 1)
                    next_wrong_xs.append(game_map.player_x + 1)

                if next_x + 1 >= game_map.x_size - 1 or game_map.player_y + 1 >= game_map.y_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y + 1)
                    next_wrong_xs.append(game_map.player_x + 1)

                if next_x >= game_map.x_size - 1 or game_map.player_y - 1 < 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y - 1)
                    next_wrong_xs.append(game_map.player_x)

                if next_x >= game_map.x_size - 1 or game_map.player_y + 1 >= game_map.y_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y + 1)
                    next_wrong_xs.append(game_map.player_x)

                if next_x >= game_map.x_size - 1:
                    action_reward += game_map.collision_cost * game_map.t_error_prob * (
                            1 / 5)
                    action_value += \
                    self.values[game_map.player_x - 1][game_map.player_y - 1][
                        game_map.player_heading] * game_map.t_error_prob * (
                                1 / 5)
                else:
                    next_wrong_ys.append(game_map.player_y)
                    next_wrong_xs.append(game_map.player_x)

            for i in range(len(next_wrong_ys)):
                this_action_reward, this_action_value = self.get_reward_value(
                    game_map, next_wrong_ys[i], next_wrong_xs[i],
                    game_map.player_heading)
                action_reward += this_action_reward * game_map.t_error_prob * (
                            1 / 5)
                action_value += this_action_value * game_map.t_error_prob * (
                            1 / 5)

        elif move == LaserTankMap.TURN_LEFT:
            # no collision or game over possible
            if game_map.player_heading == LaserTankMap.UP:
                game_map.player_heading = LaserTankMap.LEFT
            elif game_map.player_heading == LaserTankMap.DOWN:
                game_map.player_heading = LaserTankMap.RIGHT
            elif game_map.player_heading == LaserTankMap.LEFT:
                game_map.player_heading = LaserTankMap.DOWN
            else:
                game_map.player_heading = LaserTankMap.UP

            if LaserTankMap.cell_is_game_over(game_map, game_map.player_y,
                                              game_map.player_x):
                action_reward = game_map.game_over_cost
                action_value = \
                self.values[game_map.player_x - 1][game_map.player_y - 1][
                    game_map.player_heading]
            elif game_map.grid_data[game_map.player_y][
                game_map.player_x] == LaserTankMap.FLAG_SYMBOL:
                action_reward = game_map.goal_reward
                action_value = \
                self.values[game_map.player_x - 1][game_map.player_y - 1][
                    game_map.player_heading]
            else:
                action_reward = game_map.move_cost
                action_value = \
                self.values[game_map.player_x - 1][game_map.player_y - 1][
                    game_map.player_heading]


        elif move == LaserTankMap.TURN_RIGHT:
            # no collision or game over possible
            if game_map.player_heading == LaserTankMap.UP:
                game_map.player_heading = LaserTankMap.RIGHT
            elif game_map.player_heading == LaserTankMap.DOWN:
                game_map.player_heading = LaserTankMap.LEFT
            elif game_map.player_heading == LaserTankMap.LEFT:
                game_map.player_heading = LaserTankMap.UP
            else:
                game_map.player_heading = LaserTankMap.DOWN
            if LaserTankMap.cell_is_game_over(game_map, game_map.player_y,
                                              game_map.player_x):
                action_reward = game_map.game_over_cost
                action_value = \
                self.values[game_map.player_x - 1][game_map.player_y - 1][
                    game_map.player_heading]
            elif game_map.grid_data[game_map.player_y][
                game_map.player_x] == LaserTankMap.FLAG_SYMBOL:
                action_reward = game_map.goal_reward
                action_value = \
                self.values[game_map.player_x - 1][game_map.player_y - 1][
                    game_map.player_heading]
            else:
                action_reward = game_map.move_cost
                action_value = \
                self.values[game_map.player_x - 1][game_map.player_y - 1][
                    game_map.player_heading]
        # elif move == self.SHOOT_LASER:
        #     # set laser direction
        #     if map.player_heading == LaserTankMap.UP:
        #         heading = LaserTankMap.UP
        #         dy, dx = (-1, 0)
        #     elif map.player_heading == LaserTankMap.DOWN:
        #         heading = LaserTankMap.DOWN
        #         dy, dx = (1, 0)
        #     elif map.player_heading == LaserTankMap.LEFT:
        #         heading = LaserTankMap.LEFT
        #         dy, dx = (0, -1)
        #     else:
        #         heading = LaserTankMap.RIGHT
        #         dy, dx = (0, 1)
        #
        #     # loop until laser blocking object reached
        #     ly, lx = (map.player_y, map.player_x)
        #     while True:
        #         ly += dy
        #         lx += dx
        #
        #         # handle boundary and immovable obstacles
        #         if ly < 0 or ly >= map.y_size or \
        #                 lx < 0 or lx >= map.x_size or \
        #                 map.grid_data[ly][lx] == self.OBSTACLE_SYMBOL:
        #             # laser stopped without effect
        #             return map.move_cost
        #
        #         # handle movable objects
        #         elif self.cell_is_laser_movable(ly, lx, heading):
        #             # check if tile can be moved without collision
        #             if self.cell_is_blocked(ly + dy, lx + dx) or \
        #                     map.grid_data[ly + dy][
        #                         lx + dx] == LaserTankMap.ICE_SYMBOL or \
        #                     map.grid_data[ly + dy][
        #                         lx + dx] == LaserTankMap.TELEPORT_SYMBOL or \
        #                     map.grid_data[ly + dy][
        #                         lx + dx] == LaserTankMap.FLAG_SYMBOL or \
        #                     (
        #                             ly + dy == map.player_y and lx + dx == map.player_x):
        #                 # tile cannot be moved
        #                 return map.move_cost
        #             else:
        #                 old_symbol = map.grid_data[ly][lx]
        #                 map.grid_data[ly][lx] = LaserTankMap.LAND_SYMBOL
        #                 if map.grid_data[ly + dy][
        #                     lx + dx] == LaserTankMap.WATER_SYMBOL:
        #                     # if new bridge position is water, convert to land tile
        #                     if old_symbol == self.BRIDGE_SYMBOL:
        #                         map.grid_data[ly + dy][
        #                             lx + dx] = LaserTankMap.LAND_SYMBOL
        #                     # otherwise, do not replace the old symbol
        #                 else:
        #                     # otherwise, move the tile forward
        #                     map.grid_data[ly + dy][lx + dx] = old_symbol
        #                 break
        #
        #         # handle bricks
        #         elif map.grid_data[ly][lx] == self.BRICK_SYMBOL:
        #             # remove brick, replace with land
        #             map.grid_data[ly][lx] = LaserTankMap.LAND_SYMBOL
        #             break
        #
        #         # handle facing anti-tanks
        #         elif (map.grid_data[ly][
        #                   lx] == self.ANTI_TANK_UP_SYMBOL and heading == LaserTankMap.DOWN) or \
        #                 (map.grid_data[ly][
        #                      lx] == self.ANTI_TANK_DOWN_SYMBOL and heading == LaserTankMap.UP) or \
        #                 (map.grid_data[ly][
        #                      lx] == self.ANTI_TANK_LEFT_SYMBOL and heading == LaserTankMap.RIGHT) or \
        #                 (map.grid_data[ly][
        #                      lx] == self.ANTI_TANK_RIGHT_SYMBOL and heading == LaserTankMap.LEFT):
        #             # mark anti-tank as destroyed
        #             map.grid_data[ly][lx] = self.ANTI_TANK_DESTROYED_SYMBOL
        #             break
        #
        #         # handle player laser collision
        #         elif ly == map.player_y and lx == map.player_x:
        #             return map.game_over_cost
        #
        #         # handle facing mirrors
        #         elif (map.grid_data[ly][
        #                   lx] == self.MIRROR_UL_SYMBOL and heading == LaserTankMap.RIGHT) or \
        #                 (map.grid_data[ly][
        #                      lx] == self.MIRROR_UR_SYMBOL and heading == LaserTankMap.LEFT):
        #             # new direction is up
        #             dy, dx = (-1, 0)
        #             heading = LaserTankMap.UP
        #         elif (map.grid_data[ly][
        #                   lx] == self.MIRROR_DL_SYMBOL and heading == LaserTankMap.RIGHT) or \
        #                 (map.grid_data[ly][
        #                      lx] == self.MIRROR_DR_SYMBOL and heading == LaserTankMap.LEFT):
        #             # new direction is down
        #             dy, dx = (1, 0)
        #             heading = LaserTankMap.DOWN
        #         elif (map.grid_data[ly][
        #                   lx] == self.MIRROR_UL_SYMBOL and heading == LaserTankMap.DOWN) or \
        #                 (map.grid_data[ly][
        #                      lx] == self.MIRROR_DL_SYMBOL and heading == LaserTankMap.UP):
        #             # new direction is left
        #             dy, dx = (0, -1)
        #             heading = LaserTankMap.LEFT
        #         elif (map.grid_data[ly][
        #                   lx] == self.MIRROR_UR_SYMBOL and heading == LaserTankMap.DOWN) or \
        #                 (map.grid_data[ly][
        #                      lx] == self.MIRROR_DR_SYMBOL and heading == LaserTankMap.UP):
        #             # new direction is right
        #             dy, dx = (0, 1)
        #             heading = LaserTankMap.RIGHT
        #         # do not terminate laser on facing mirror - keep looping
        #
        #     # check for game over condition after effect of laser
        #     if self.cell_is_game_over(map.player_y, map.player_x):
        #         return map.game_over_cost

        return action_reward + action_value * game_map.gamma

if __name__ == "__main__":
    grid = Grid
    vi = ValueIteration(grid)

    start = time.time()
    print("Initial values:")
    vi.print_values()
    print()

    for i in range(max_iter):
        vi.next_iteration()
        print("Values after iteration", i + 1)
        vi.print_values()
        print()

    end = time.time()
    print("Time to copmlete", max_iter, "VI iterations")
    print(end - start)
    