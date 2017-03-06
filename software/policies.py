from simulation_enviroment import Cell
import abc
import random
from math import floor, log
import itertools


class BasePolicy():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, reward_function):
        self.rewards = []
        self.reward_function = reward_function

    @abc.abstractmethod
    def policy(self, s, cells):
        return [Cell.off for _ in cells]

    @abc.abstractmethod
    def reward(self, s, cells):
        return self.reward_function(s, cells)



class LatestSwitchPolicy(BasePolicy):

    def policy(self, s, cells):
        cell_policies = []
        for cell in cells:
            if cell.temperature >= cell.max_temp:
                cell_policies.append(Cell.on)

            elif cell.temperature <= cell.min_temp:
                cell_policies.append(Cell.off)

            else:
                cell_policies.append(cell.state)

        return cell_policies

class QLearningPolicy(BasePolicy):

    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qvalues = {}
        self.legal_actions = None

    def set_legal_actions(self, cells):
        """
        CAUTION!, factorial memory complexity
        """
        all_actions = [[0,1] for _ in cells]
        self.legal_actions = list(itertools.product(*all_actions))

    def policy(self, s, cells):
        if not self.legal_actions:
            self.set_legal_actions(cells)
        if random.random() <= self.epsilon:
            return list(random.choice(self.legal_actions))
        tile = self.tile_state(s, cells)
        return list(self.compute_action_from_q(tile))

    def get_q_value(self, tile, action):
        if not isinstance(action, tuple):
            action = tuple(action)

        if (tile, action) not in self.qvalues:
            self.qvalues[(tile, action)] = 0.0

        return self.qvalues[(tile, action)]

    def compute_value_from_q(self, tile):
        actions = self.legal_actions
        max_val = -float('Inf')
        for action in actions:
            qval = self.get_q_value(tile, action)

            if qval > max_val:
                max_q_val = qval

        return max_q_val

    def compute_action_from_q(self, tile):
        actions = self.legal_actions
        max_q_val = -float('Inf')
        best_action = None
        for action in actions:
            qval = self.get_q_value(tile, action)

            if qval > max_q_val:
                max_q_val = qval
                best_action = action

        return best_action


    def tile_state(self, s, cells):
        tile = []
        for cell in cells:
            temp = cell.temperature
            tile.append(cell.state)
            if  cell.min_temp < temp < cell.max_temp:
                tiled_temp = int(floor(temp * 10))
            elif temp < cell.min_temp:
                tiled_temp = -100
            elif temp > cell.max_temp:
                tiled_temp = 100
            tile.append(tiled_temp)
        realised = s['pricing'] * 10
        realised = int(floor(realised))
        tile.append(realised)
        return tuple(tile)

    def update(self, s, cells, action, next_sate, next_sate_cells, reward):
        if not isinstance(action, tuple):
            action = tuple(action)
        if not self.legal_actions:
            self.set_legal_actions(cells)
        tile = self.tile_state(s, cells)
        next_tile = self.tile_state(next_sate, next_sate_cells)
        # update
        self.qvalues[(tile, action)] = (
            self.get_q_value(tile, action) + self.alpha * (
            reward + self.gamma * self.compute_value_from_q(next_tile) - self.get_q_value(tile, action)
            )
        )
