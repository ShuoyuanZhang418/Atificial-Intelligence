from laser_tank import LaserTankMap, DotDict
import random
import time
import numpy as np
import matplotlib.pyplot as plt
"""
Template file for you to implement your solution to Assignment 4. You should implement your solution by filling in the
following method stubs:
    train_q_learning()
    train_sarsa()
    get_policy()
    
You may add to the __init__ method if required, and can add additional helper methods and classes if you wish.

To ensure your code is handled correctly by the autograder, you should avoid using any try-except blocks in your
implementation of the above methods (as this can interfere with our time-out handling).

COMP3702 2020 Assignment 4 Support Code
"""


class Solver:

    def __init__(self, a):
        """
        Initialise solver without a Q-value table.
        """
        self.epsilon = 0.6
        self.learning_rate = a
        self.avg_reward = []
        #
        # TODO
        # You may add code here if you wish (e.g. define constants used by both methods).
        #
        # The allowed time for this method is 1 second.
        #

        self.q_values = None

    def choose_action(self, q_values, state, actions):
        current_state = (state.player_x, state.player_y, state.player_heading)
        if np.random.uniform() > self.epsilon:
            values = []
            for action in actions:
                values.append(q_values[hash(current_state)][action])
            a = actions[np.argmax(values)]
        else:
            a = np.random.choice(actions)
        return a

    def train_q_learning(self, simulator):
        """
        Train the agent using Q-learning, building up a table of Q-values.
        :param simulator: A simulator for collecting episode data (LaserTankMap instance)
        """
        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        params = simulator.params
        gamma = simulator.params.gamma
        time_limit = simulator.params.time_limit
        actions = simulator.MOVES
        states = list((x, y, d) for x in range(params.x_size) for y in range(params.y_size) for d in range(4))
        q_values = {hash(state): {action: 0 for action in actions} for state in states}
        start_time = time.time()
        n = 0
        total_reward = 0
        avg_reward = []
        while n < 2000:
            episode = False
            simulator.reset_to_start()
            while not episode:
                action = self.choose_action(q_values, simulator, actions)
                current_state = simulator.make_clone()
                s = (current_state.player_x, current_state.player_y, current_state.player_heading)
                q_value = q_values[hash(s)][action]
                reward, episode = simulator.apply_move(action)
                total_reward += reward
                s0 = (simulator.player_x, simulator.player_y, simulator.player_heading)
                values = []
                for a in actions:
                    values.append(q_values[hash(s0)][a])
                max_q_value = np.max(values)
                if not episode:
                    new_q_value = q_value + self.learning_rate * (reward + gamma * max_q_value - q_value)
                else:
                    new_q_value = reward
                q_values[hash(s)][action] = new_q_value
            n += 1
            if n % 50 == 0:
                avg_reward.append(total_reward / 50)
                total_reward = 0
            run_time = time.time() - start_time
            if run_time >= time_limit:
                break
        #
        # TODO
        # Write your Q-Learning implementation here.
        #
        # When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should
        # contnue training until the time limit is reached.
        #
        # store the computed Q-values
        print(avg_reward)
        self.avg_reward = avg_reward
        self.q_values = q_values

    def train_sarsa(self, simulator):
        """
        Train the agent using SARSA, building up a table of Q-values.
        :param simulator: A simulator for collecting episode data (LaserTankMap instance)
        """

        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        params = simulator.params
        gamma = simulator.params.gamma
        time_limit = simulator.params.time_limit
        actions = simulator.MOVES
        states = list((x, y, d) for x in range(params.x_size) for y in range(params.y_size) for d in range(4))
        q_values = {hash(state): {action: 0 for action in actions} for state in states}
        start_time = time.time()
        avg_reward = []
        n = 0
        total_reward = 0
        while True:
            episode = False
            simulator.reset_to_start()
            next_action = self.choose_action(q_values, simulator, actions)
            while not episode:
                action = next_action
                current_state = simulator.make_clone()
                s = (current_state.player_x, current_state.player_y, current_state.player_heading)
                q_value = q_values[hash(s)][action]
                reward, episode = simulator.apply_move(action)
                total_reward += reward
                s0 = (simulator.player_x, simulator.player_y, simulator.player_heading)
                next_action = self.choose_action(q_values, simulator, actions)
                next_q_value = q_values[hash(s0)][next_action]
                if not episode:
                    new_q_value = q_value + self.learning_rate * (reward + gamma * next_q_value - q_value)
                else:
                    new_q_value = reward
                q_values[hash(s)][action] = new_q_value
            n += 1
            if n % 50 == 0:
                avg_reward.append(total_reward / 50)
                total_reward = 0
            run_time = time.time() - start_time
            if run_time >= time_limit:
                break

        #
        # TODO
        # Write your SARSA implementation here.
        #
        # When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should
        # continue training until the time limit is reached.
        #

        # store the computed Q-values
        print(avg_reward)
        self.avg_reward = avg_reward
        self.q_values = q_values


    def get_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """
        actions = ['f', 'l', 'r', 's']
        values = []
        current_state = (state.player_x, state.player_y, state.player_heading)
        for action in actions:
            values.append(self.q_values[hash(current_state)][action])
        policy = actions[np.argmax(values)]
        return policy
        #
        # TODO
        # Write code to return the optimal action to be performed at this state based on the stored Q-values.
        #
        # You can assume that either train_q_learning( ) or train_sarsa( ) has been called before this
        # method is called.
        #
        # When this method is called, you are allowed up to 1 second of compute time.
        #

        pass

    def f(self, t):
        #if t == 0:
            #return self.avg_reward[t]
        #else:
        a = t
        b = np.array(self.avg_reward)
        return np.array(self.avg_reward)[a]


if __name__ == '__main__':
    game_map = LaserTankMap.process_input_file("testcases/q-learn_t1.txt")
    game_map2 = LaserTankMap.process_input_file("testcases/sarsa_t1.txt")
    simulator = game_map.make_clone()
    simulator2 = game_map2.make_clone()
    a1 = 0.05
    a2 = 0.1
    a3 = 0.5
    solver1 = Solver(a1)
    solver1.train_q_learning(simulator)
    solver2 = Solver(a1)
    solver2.train_q_learning(simulator2)
    #solver2 = Solver(a2)
    #solver2.train_q_learning(simulator)
    #solver3 = Solver(a3)
    #solver3.train_q_learning(simulator)
    t = [i*50 for i in range(len(solver1.avg_reward))]
    #plt.title("The quality of policy learned by q-learning by different learning rate")
    plt.title("The quality of policy learned by q-learning and SARSA")
    plt.xlabel('Episodes')
    plt.ylabel('50-step average reward')
    plt.plot(t, solver1.avg_reward, label='q-leaning')
    plt.plot(t, solver2.avg_reward, label='SARSA')
    #plt.plot(t, solver2.avg_reward, label='alpha = 0.1')
    #plt.plot(t, solver3.avg_reward, label='alpha = 0.5')
    plt.legend(['q-leaning', 'SARSA'])
    #plt.legend(['alpha = 0.05', 'alpha = 0.1', 'alpha = 0.5'])
    #plt.plot(t, solver1.avg_reward, 'rs-',
             #t, solver2.avg_reward, 'bs-',
             #t, solver3.avg_reward, 'gs-')
    plt.legend()
    plt.show()







