from simulation_enviroment import Cell
import abc

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

    def __init__(self, reward_function):
        super(LatestSwitchPolicy, self).__init__(reward_function)

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

    def __init__(self, alpha, gamma, reward_function):
        pass

    def policy(self, s, cells):
        pass
