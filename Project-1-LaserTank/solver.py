#!/usr/bin/python
import heapq
import math
import sys





"""
Template file for you to implement your solution to Assignment 1.

COMP3702 2020 Assignment 1 Support Code
"""


#
#
# Code for any classes or functions you need can go here.
#
#
"""
PriorityQueue class reference from https://github.com/aahuja9/Pacman-AI.
"""
class PriorityQueue:

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        # FIXME: restored old behaviour to check against old results better
        # FIXED: restored to stable behaviour
        entry = (priority, self.count, item)
        # entry = (priority, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        #  (_, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0





class LaserTankMap:
    """
    Instance of a LaserTank game map. You may use this class and its functions
    directly or duplicate and modify it in your solution. To ensure Tester
    functions correctly, you should avoid modifying this file directly.
    """

    # input file symbols
    LAND_SYMBOL = ' '
    WATER_SYMBOL = 'W'
    OBSTACLE_SYMBOL = '#'
    BRIDGE_SYMBOL = 'B'
    BRICK_SYMBOL = 'K'
    ICE_SYMBOL = 'I'
    TELEPORT_SYMBOL = 'T'
    FLAG_SYMBOL = 'F'

    MIRROR_UL_SYMBOL = '1'
    MIRROR_UR_SYMBOL = '2'
    MIRROR_DL_SYMBOL = '3'
    MIRROR_DR_SYMBOL = '4'

    PLAYER_UP_SYMBOL = '^'  # note: player always starts on a land tile
    PLAYER_DOWN_SYMBOL = 'v'
    PLAYER_LEFT_SYMBOL = '<'
    PLAYER_RIGHT_SYMBOL = '>'

    ANTI_TANK_UP_SYMBOL = 'U'
    ANTI_TANK_DOWN_SYMBOL = 'D'
    ANTI_TANK_LEFT_SYMBOL = 'L'
    ANTI_TANK_RIGHT_SYMBOL = 'R'
    ANTI_TANK_DESTROYED_SYMBOL = 'X'

    VALID_SYMBOLS = [LAND_SYMBOL, WATER_SYMBOL, OBSTACLE_SYMBOL, BRIDGE_SYMBOL, BRICK_SYMBOL, ICE_SYMBOL,
                     TELEPORT_SYMBOL, FLAG_SYMBOL, MIRROR_UL_SYMBOL, MIRROR_UR_SYMBOL, MIRROR_DL_SYMBOL,
                     MIRROR_DR_SYMBOL, PLAYER_UP_SYMBOL, PLAYER_DOWN_SYMBOL, PLAYER_LEFT_SYMBOL, PLAYER_RIGHT_SYMBOL,
                     ANTI_TANK_UP_SYMBOL, ANTI_TANK_DOWN_SYMBOL, ANTI_TANK_LEFT_SYMBOL, ANTI_TANK_RIGHT_SYMBOL,
                     ANTI_TANK_DESTROYED_SYMBOL]

    # move symbols (i.e. output file symbols)
    MOVE_FORWARD = 'f'
    TURN_LEFT = 'l'
    TURN_RIGHT = 'r'
    SHOOT_LASER = 's'
    MOVES = [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, SHOOT_LASER]

    # directions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # move return statuses
    SUCCESS = 0
    COLLISION = 1
    GAME_OVER = 2

    # render characters
    MAP_GLYPH_TABLE = {LAND_SYMBOL: '   ', WATER_SYMBOL: 'WWW', OBSTACLE_SYMBOL: 'XXX', BRIDGE_SYMBOL: '[B]',
                       BRICK_SYMBOL: '[K]', ICE_SYMBOL: '-I-', TELEPORT_SYMBOL: '(T)', FLAG_SYMBOL: ' F ',
                       MIRROR_UL_SYMBOL: ' /|', MIRROR_UR_SYMBOL: '|\\ ', MIRROR_DL_SYMBOL: ' \\|',
                       MIRROR_DR_SYMBOL: '|/ ', ANTI_TANK_UP_SYMBOL: '[U]', ANTI_TANK_DOWN_SYMBOL: '[D]',
                       ANTI_TANK_LEFT_SYMBOL: '[L]', ANTI_TANK_RIGHT_SYMBOL: '[R]', ANTI_TANK_DESTROYED_SYMBOL: '[X]'}
    PLAYER_GLYPH_TABLE = {UP: '[^]', DOWN: '[v]', LEFT: '[<]', RIGHT: '[>]'}

    # symbols which are movable for each direction
    MOVABLE_SYMBOLS = {UP: [BRIDGE_SYMBOL, MIRROR_UL_SYMBOL, MIRROR_UR_SYMBOL, ANTI_TANK_UP_SYMBOL,
                            ANTI_TANK_LEFT_SYMBOL, ANTI_TANK_RIGHT_SYMBOL, ANTI_TANK_DESTROYED_SYMBOL],
                       DOWN: [BRIDGE_SYMBOL, MIRROR_DL_SYMBOL, MIRROR_DR_SYMBOL, ANTI_TANK_DOWN_SYMBOL,
                              ANTI_TANK_LEFT_SYMBOL, ANTI_TANK_RIGHT_SYMBOL, ANTI_TANK_DESTROYED_SYMBOL],
                       LEFT: [BRIDGE_SYMBOL, MIRROR_UL_SYMBOL, MIRROR_DL_SYMBOL, ANTI_TANK_UP_SYMBOL,
                              ANTI_TANK_DOWN_SYMBOL, ANTI_TANK_LEFT_SYMBOL, ANTI_TANK_DESTROYED_SYMBOL],
                       RIGHT: [BRIDGE_SYMBOL, MIRROR_UR_SYMBOL, MIRROR_DR_SYMBOL, ANTI_TANK_UP_SYMBOL,
                               ANTI_TANK_DOWN_SYMBOL, ANTI_TANK_RIGHT_SYMBOL, ANTI_TANK_DESTROYED_SYMBOL]
                       }

    def __init__(self, x_size, y_size, grid_data, player_x=None, player_y=None, player_heading=None, flag_x=None, flag_y=None):
        """
        Build a LaserTank map instance from the given grid data.
        :param x_size: width of map
        :param y_size: height of map
        :param grid_data: matrix with dimensions (y_size, x_size) where each element is a valid symbol
        """
        self.x_size = x_size
        self.y_size = y_size
        self.grid_data = grid_data

        for i in range(y_size):
            row = self.grid_data[i]
            for j in range(x_size):
                if row[j] == self.FLAG_SYMBOL:
                    self.flag_x = j
                    self.flag_y = i
                    break

        # extract player position and heading if none given
        if player_x is None and player_y is None and player_heading is None:
            found = False
            for i in range(y_size):
                row = self.grid_data[i]
                for j in range(x_size):
                    if row[j] == self.PLAYER_UP_SYMBOL or row[j] == self.PLAYER_DOWN_SYMBOL or \
                            row[j] == self.PLAYER_LEFT_SYMBOL or row[j] == self.PLAYER_RIGHT_SYMBOL:
                        found = True
                        self.player_x = j
                        self.player_y = i
                        self.player_heading = {self.PLAYER_UP_SYMBOL: self.UP,
                                               self.PLAYER_DOWN_SYMBOL: self.DOWN,
                                               self.PLAYER_LEFT_SYMBOL: self.LEFT,
                                               self.PLAYER_RIGHT_SYMBOL: self.RIGHT}[row[j]]
                        # replace the player symbol with land tile
                        row[j] = self.LAND_SYMBOL
                        break
                if found:
                    break
            if not found:
                raise Exception("LaserTank Map Error: Grid data does not contain player symbol")
        elif player_x is None or player_y is None or player_heading is None:
            raise Exception("LaserTank Map Error: Incomplete player coordinates given")
        else:
            self.player_x = player_x
            self.player_y = player_y
            self.player_heading = player_heading


    @staticmethod
    def process_input_file(filename):
        """
        Process the given input file and create a new map instance based on the input file.
        :param filename: name of input file
        """
        f = open(filename, 'r')

        rows = []
        i = 0
        for line in f:
            # skip optimal steps and time limit
            if i > 1 and len(line.strip()) > 0:
                rows.append(list(line.strip()))
            i += 1

        f.close()

        row_len = len(rows[0])
        num_rows = len(rows)

        return LaserTankMap(row_len, num_rows, rows)

    def apply_move(self, move, state):
        """
        Apply a player move to the map.
        :param move: self.MOVE_FORWARD, self.TURN_LEFT, self.TURN_RIGHT or self.SHOOT_LASER
        :return: LaserTankMap.SUCCESS if move was successful and no collision (or no effect move) occurred,
                 LaserTankMap.COLLISION if the move resulted collision or had no effect,
                 LaserTankMap.GAME_OVER if the move resulted in game over
        """
        x, y , heading, grid_data = state
        map_data = [row[:] for row in grid_data]
        if move == self.MOVE_FORWARD:
            # get coordinates for next cell
            if heading == self.UP:
                next_y = y - 1
                next_x = x
            elif heading == self.DOWN:
                next_y = y + 1
                next_x = x
            elif heading == self.LEFT:
                next_y = y
                next_x = x - 1
            else:
                next_y = y
                next_x = x + 1

            # handle special tile types
            if map_data[next_y][next_x] == self.ICE_SYMBOL:
                # handle ice tile - slide until first non-ice tile or blocked
                if heading == self.UP:
                    for i in range(next_y, -1, -1):
                        if map_data[i][next_x] != self.ICE_SYMBOL:
                            if map_data[i][next_x] == self.WATER_SYMBOL:
                                # slide into water - game over
                                return self.GAME_OVER
                            elif self.cell_is_blocked(i, next_x, map_data):
                                # if blocked, stop on last ice cell
                                next_y = i + 1
                                break
                            else:
                                next_y = i
                                break
                elif heading == self.DOWN:
                    for i in range(next_y, self.y_size):
                        if map_data[i][next_x] != self.ICE_SYMBOL:
                            if map_data[i][next_x] == self.WATER_SYMBOL:
                                # slide into water - game over
                                return self.GAME_OVER
                            elif self.cell_is_blocked(i, next_x, map_data):
                                # if blocked, stop on last ice cell
                                next_y = i - 1
                                break
                            else:
                                next_y = i
                                break
                elif heading == self.LEFT:
                    for i in range(next_x, -1, -1):
                        if map_data[next_y][i] != self.ICE_SYMBOL:
                            if map_data[next_y][i] == self.WATER_SYMBOL:
                                # slide into water - game over
                                return self.GAME_OVER
                            elif self.cell_is_blocked(next_y, i, map_data):
                                # if blocked, stop on last ice cell
                                next_x = i + 1
                                break
                            else:
                                next_x = i
                                break
                else:
                    for i in range(next_x, self.x_size):
                        if map_data[next_y][i] != self.ICE_SYMBOL:
                            if map_data[next_y][i] == self.WATER_SYMBOL:
                                # slide into water - game over
                                return self.GAME_OVER
                            elif self.cell_is_blocked(next_y, i, map_data):
                                # if blocked, stop on last ice cell
                                next_x = i - 1
                                break
                            else:
                                next_x = i
                                break
            if map_data[next_y][next_x] == self.TELEPORT_SYMBOL:
                # handle teleport - find the other teleporter
                tpy, tpx = (None, None)
                for i in range(self.y_size):
                    for j in range(self.x_size):
                        if map_data[i][j] == self.TELEPORT_SYMBOL and (i != next_y or j != next_x):
                            tpy, tpx = (i, j)
                            break
                    if tpy is not None:
                        break
                if tpy is None:
                    raise Exception("LaserTank Map Error: Unmatched teleport symbol")
                next_y, next_x = (tpy, tpx)
            else:
                # if not ice or teleport, perform collision check
                if self.cell_is_blocked(next_y, next_x, map_data):
                    return self.COLLISION

            # check for game over conditions
            if self.cell_is_game_over(next_y, next_x, map_data):
                return self.GAME_OVER

            # no collision and no game over - update player position
            y = next_y
            x = next_x
            return (x, y, heading, map_data)

        elif move == self.TURN_LEFT:
            # no collision or game over possible
            if heading == self.UP:
                heading = self.LEFT
            elif heading == self.DOWN:
                heading = self.RIGHT
            elif heading == self.LEFT:
                heading = self.DOWN
            else:
                heading = self.UP
            return (x, y, heading, map_data)

        elif move == self.TURN_RIGHT:
            # no collision or game over possible
            if heading == self.UP:
                heading = self.RIGHT
            elif heading == self.DOWN:
                heading = self.LEFT
            elif heading == self.LEFT:
                heading = self.UP
            else:
                heading = self.DOWN
            return (x, y, heading, map_data)

        elif move == self.SHOOT_LASER:
            # set laser direction
            if heading == self.UP:
                laserheading = self.UP
                dy, dx = (-1, 0)
            elif heading == self.DOWN:
                laserheading = self.DOWN
                dy, dx = (1, 0)
            elif heading == self.LEFT:
                laserheading = self.LEFT
                dy, dx = (0, -1)
            else:
                laserheading = self.RIGHT
                dy, dx = (0, 1)

            # loop until laser blocking object reached
            ly, lx = (y, x)
            while True:
                ly += dy
                lx += dx

                # handle boundary and immovable obstacles
                if ly < 0 or ly >= self.y_size or \
                        lx < 0 or lx >= self.x_size or \
                        map_data[ly][lx] == self.OBSTACLE_SYMBOL:
                    # laser stopped without effect
                    return self.COLLISION

                # handle movable objects
                elif self.cell_is_laser_movable(ly, lx, laserheading, map_data):
                    # check if tile can be moved without collision
                    if self.cell_is_blocked(ly + dy, lx + dx, map_data) or \
                            map_data[ly + dy][lx + dx] == self.ICE_SYMBOL or \
                            map_data[ly + dy][lx + dx] == self.TELEPORT_SYMBOL or \
                            map_data[ly + dy][lx + dx] == self.FLAG_SYMBOL or \
                            (ly + dy == y and lx + dx == x):
                        # tile cannot be moved
                        return self.COLLISION
                    else:
                        old_symbol = map_data[ly][lx]
                        map_data[ly][lx] = self.LAND_SYMBOL
                        if map_data[ly + dy][lx + dx] == self.WATER_SYMBOL:
                            # if new bridge position is water, convert to land tile
                            if old_symbol == self.BRIDGE_SYMBOL:
                                map_data[ly + dy][lx + dx] = self.LAND_SYMBOL
                            # otherwise, do not replace the old symbol
                        else:
                            # otherwise, move the tile forward
                            map_data[ly + dy][lx + dx] = old_symbol
                        break

                # handle bricks
                elif map_data[ly][lx] == self.BRICK_SYMBOL:
                    # remove brick, replace with land
                    map_data[ly][lx] = self.LAND_SYMBOL
                    break

                # handle facing anti-tanks
                elif (map_data[ly][lx] == self.ANTI_TANK_UP_SYMBOL and laserheading == self.DOWN) or \
                        (map_data[ly][lx] == self.ANTI_TANK_DOWN_SYMBOL and laserheading == self.UP) or \
                        (map_data[ly][lx] == self.ANTI_TANK_LEFT_SYMBOL and laserheading == self.RIGHT) or \
                        (map_data[ly][lx] == self.ANTI_TANK_RIGHT_SYMBOL and laserheading == self.LEFT):
                    # mark anti-tank as destroyed
                    map_data[ly][lx] = self.ANTI_TANK_DESTROYED_SYMBOL
                    break

                # handle player laser collision
                elif ly == y and lx == x:
                    return self.GAME_OVER

                # handle facing mirrors
                elif (map_data[ly][lx] == self.MIRROR_UL_SYMBOL and laserheading == self.RIGHT) or \
                        (map_data[ly][lx] == self.MIRROR_UR_SYMBOL and laserheading == self.LEFT):
                    # new direction is up
                    dy, dx = (-1, 0)
                    laserheading = self.UP
                elif (map_data[ly][lx] == self.MIRROR_DL_SYMBOL and laserheading == self.RIGHT) or \
                        (self.grid_data[ly][lx] == self.MIRROR_DR_SYMBOL and laserheading == self.LEFT):
                    # new direction is down
                    dy, dx = (1, 0)
                    laserheading = self.DOWN
                elif (map_data[ly][lx] == self.MIRROR_UL_SYMBOL and laserheading == self.DOWN) or \
                        (map_data[ly][lx] == self.MIRROR_DL_SYMBOL and laserheading == self.UP):
                    # new direction is left
                    dy, dx = (0, -1)
                    laserheading = self.LEFT
                elif (map_data[ly][lx] == self.MIRROR_UR_SYMBOL and laserheading == self.DOWN) or \
                        (map_data[ly][lx] == self.MIRROR_DR_SYMBOL and laserheading == self.UP):
                    # new direction is right
                    dy, dx = (0, 1)
                    laserheading = self.RIGHT
                # do not terminate laser on facing mirror - keep looping

            # check for game over condition after effect of laser
            if self.cell_is_game_over(y, x, map_data):
                return self.GAME_OVER
            return (x, y, heading, map_data)
        return self.SUCCESS

    def render(self):
        """
        Render the map's current state to terminal
        """
        for r in range(self.y_size):
            line = ''
            for c in range(self.x_size):
                glyph = self.MAP_GLYPH_TABLE[self.grid_data[r][c]]

                # overwrite with player
                if r == self.player_y and c == self.player_x:
                    glyph = self.PLAYER_GLYPH_TABLE[self.player_heading]

                line += glyph
            print(line)

        print('\n' * (20 - self.y_size))

    def is_finished(self, state):
        """
        Check if the finish condition (player at flag) has been reached
        :return: True if player at flag, False otherwise
        """
        x, y, heading, map_data = state
        if map_data[y][x] == self.FLAG_SYMBOL:
            return True
        else:
            return False

    def cell_is_blocked(self, y, x, map_data):
        """
        Check if the cell with the given coordinates is blocked (i.e. movement
        to this cell is not possible)
        :param y: y coord
        :param x: x coord
        :return: True if blocked, False otherwise
        """
        symbol = map_data[y][x]
        # collision: obstacle, bridge, mirror (all types), anti-tank (all types)
        if symbol == self.OBSTACLE_SYMBOL or symbol == self.BRIDGE_SYMBOL or symbol == self.BRICK_SYMBOL or \
                symbol == self.MIRROR_UL_SYMBOL or symbol == self.MIRROR_UR_SYMBOL or \
                symbol == self.MIRROR_DL_SYMBOL or symbol == self.MIRROR_DR_SYMBOL or \
                symbol == self.ANTI_TANK_UP_SYMBOL or symbol == self.ANTI_TANK_DOWN_SYMBOL or \
                symbol == self.ANTI_TANK_LEFT_SYMBOL or symbol == self.ANTI_TANK_RIGHT_SYMBOL or \
                symbol == self.ANTI_TANK_DESTROYED_SYMBOL:
            return True
        return False

    def cell_is_game_over(self, y, x, map_data):
        """
        Check if the cell with the given coordinates will result in game
        over.
        :param y: y coord
        :param x: x coord
        :return: True if blocked, False otherwise
        """
        # check for water
        if map_data[y][x] == self.WATER_SYMBOL:
            return True

        # check for anti-tank
        # up direction
        for i in range(y, -1, -1):
            if map_data[i][x] == self.ANTI_TANK_DOWN_SYMBOL:
                return True
            # if blocked, can stop checking for anti-tank
            if self.cell_is_blocked(i, x, map_data):
                break

        # down direction
        for i in range(y, self.y_size):
            if map_data[i][x] == self.ANTI_TANK_UP_SYMBOL:
                return True
            # if blocked, can stop checking for anti-tank
            if self.cell_is_blocked(i, x, map_data):
                break

        # left direction
        for i in range(x, -1, -1):
            if map_data[y][i] == self.ANTI_TANK_RIGHT_SYMBOL:
                return True
            # if blocked, can stop checking for anti-tank
            if self.cell_is_blocked(y, i, map_data):
                break

        # right direction
        for i in range(x, self.x_size):
            if map_data[y][i] == self.ANTI_TANK_LEFT_SYMBOL:
                return True
            # if blocked, can stop checking for anti-tank
            if self.cell_is_blocked(y, i, map_data):
                break

        # no water or anti-tank danger
        return False

    def cell_is_laser_movable(self, y, x, heading, map_data):
        """
        Check if the tile at coordinated (y, x) is movable by a laser with the given heading.
        :param y: y coord
        :param x: x coord
        :param heading: laser direction
        :return: True is movable, false otherwise
        """
        return map_data[y][x] in self.MOVABLE_SYMBOLS[heading]

    def getStartState(self):
        return (self.player_x, self.player_y, self.player_heading, self.grid_data)

    def getNextstate(self,state):
        successors = []
        for action in [self.MOVE_FORWARD, self.TURN_LEFT, self.TURN_RIGHT, self.SHOOT_LASER]:
            if self.apply_move(action, state) != self.GAME_OVER and self.apply_move(action, state) != self.COLLISION:
                nextState = self.apply_move(action, state)
                successors.append((nextState, action))
        return successors

    def getOdistance(self, x, y):
        return max(abs(x - self.flag_x), abs(y - self.flag_y))

    def getQdistance(self, x, y):
        return math.sqrt((x - self.flag_x)**2 + (y - self.flag_y)**2)

    def getCostOfActions(self, actions):
        cost = 0
        for action in actions:
            cost += 1
        return cost

    def getMdistance(self, x, y):
        return abs(x - self.flag_x) + abs(y - self.flag_y)

    def heuristic(self, state, action):
        x1, y1, heading1, map_data1 = self.getStartState()
        x, y, heading, map_data = state
        heu = 0
        #distance = self.getQdistance(x, y)
        #distance = self.getOdistance(x, y)
        if len(action) < 3:
            heu = 0
        else:
            distance = self.getMdistance(x, y)
            heu += distance
        return heu

    def getHashstate(self, state):
        x, y ,heading, grid_data = state
        return hash((x, y, heading) + tuple([item for sublist in grid_data for item in sublist]))
"""
UCS and A* searching algorithm design reference from https://github.com/aahuja9/Pacman-AI.
Instead of using the parent node to return the final result, this approach stores the state 
directly into the priority queue along with the sequence of legal actions arriving at the 
state, which I think is a good idea to reduce the amount of code and elapsed time.
"""

def uniformCostSearch(LaserTankMap):
    start = LaserTankMap.getStartState()
    exploredState = set()
    states = PriorityQueue()
    states.push((start, []), 0)
    while not states.isEmpty():
        state, actions = states.pop()
        if LaserTankMap.is_finished(state):
            return actions
        if LaserTankMap.getHashstate(state) not in exploredState:
            nextStates = LaserTankMap.getNextstate(state)
            for next in nextStates:
                coordinates = next[0]
                if LaserTankMap.getHashstate(coordinates) not in exploredState:
                    directions = next[1]
                    totalCost = actions + [directions]
                    states.push((coordinates, actions + [directions]), LaserTankMap.getCostOfActions(totalCost))
        exploredState.add(LaserTankMap.getHashstate(state))
    return actions




def aStarSearch(LaserTankMap):
    start = LaserTankMap.getStartState()
    exploredState = set()
    states = PriorityQueue()
    states.push((start, []), 0)
    while not states.isEmpty():
        state, actions = states.pop()
        if LaserTankMap.is_finished(state):
            return actions
        if LaserTankMap.getHashstate(state) not in exploredState:
            nextStates = LaserTankMap.getNextstate(state)
            for next in nextStates:
                coordinates = next[0]
                if LaserTankMap.getHashstate(coordinates) not in exploredState:
                    directions = next[1]
                    totalActions = actions + [directions]
                    totalCost = LaserTankMap.getCostOfActions(totalActions) + LaserTankMap.heuristic(coordinates, totalActions)
                    states.push((coordinates, actions + [directions]), totalCost)
        exploredState.add(LaserTankMap.getHashstate(state))
    return actions


def write_output_file(filename, actions):
    """
    Write a list of actions to an output file. You should use this method to write your output file.
    :param filename: name of output file
    :param actions: list of actions where is action is in LaserTankMap.MOVES
    """
    f = open(filename, 'w')
    for i in range(len(actions)):
        f.write(str(actions[i]))
        if i < len(actions) - 1:
            f.write(',')
    f.write('\n')
    f.close()


def main(arglist):
    input_file = arglist[0]
    output_file = arglist[1]

    # Read the input testcase file
    game_map = LaserTankMap.process_input_file(input_file)

    #actions = uniformCostSearch(game_map)
    actions = aStarSearch(game_map)
    #
    #
    # Code for your main method can go here.
    #
    # Your code should find a sequence of actions for the agent to follow to reach the goal, and store this sequence
    # in 'actions'.
    #
    #
    print(actions)
    # Write the solution to the output file
    write_output_file(output_file, actions)


if __name__ == '__main__':
    main(["testcases/t3_the_river.txt", "ti1.txt"])
    #main(sys.argv[1:])


