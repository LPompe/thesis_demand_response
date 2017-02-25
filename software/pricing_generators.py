
import matplotlib.pyplot as plt
import numpy as np
from random import randint, choice
import pandas as pd
import abc

class BasePricingGenerator():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, length):
        self.length = length

    @abc.abstractmethod
    def generate_episode_pricing(self):
        return np.linspace(1, 1, self.length)

class StaticSinePricingGenerator(BasePricingGenerator):

    def generate_episode_pricing(self):
        linspace = np.linspace(0, self.length, self.length)
        f = 0.130900 * 2
        sine = np.cos(linspace * f) / 2 + 0.5
        return sine

class ShiftSinePricingGenerator(BasePricingGenerator):

    def generate_episode_pricing(self):
        linspace = np.linspace(0, self.length, self.length)
        f = 0.130900 * 2
        shift = randint(0, self.length)
        sine = np.cos(linspace * f + shift) / 2 + 0.5
        return sine

class ApxStaticPricingGenerator(BasePricingGenerator):

    def __init__(self, length):
        super(ApxStaticPricingGenerator, self).__init__(length)
        dataset = pd.read_csv('data/APX.csv', index_col=0)
        dataset = dataset['2016-01-04':]['APX'] # start on a monday
        min_ , max_ = dataset.min(), dataset.max()
        self.dataset = (dataset - min_).divide(max_ - min_)
        self.last_index = 0

    def generate_episode_pricing(self):
        start = self.last_index
        end = start + self.length
        self.last_index = end
        if end < len(self.dataset):
            return list(self.dataset.iloc[start:end])
        else:
            # wrap
            self.last_index = end - len(self.dataset)
            end = self.last_index
            return list(self.dataset.iloc[start:]) + list(self.dataset.iloc[:end])


class ApxShiftPricingGenerator(BasePricingGenerator):

    def __init__(self, length):
        super(ApxShiftPricingGenerator, self).__init__(length)
        dataset = pd.read_csv('data/APX.csv', index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        dataset = dataset['2016-01-04':]['APX'] # start on a monday
        min_ , max_ = dataset.min(), dataset.max()
        self.dataset = (dataset - min_).divide(max_ - min_)
        mondays = pd.date_range(start = '2016-01-04', end='2017-01-01', freq='7D')
        start_point = choice(mondays)
        self.dataset = pd.concat([self.dataset[start_point:], self.dataset], axis=0)
        self.last_index = 0

    def generate_episode_pricing(self):
        start = self.last_index
        end = start + self.length
        self.last_index = end
        if end < len(self.dataset):
            return list(self.dataset.iloc[start:end])
        else:
            # wrap
            self.last_index = end - len(self.dataset)
            end = self.last_index
            return list(self.dataset.iloc[start:]) + list(self.dataset.iloc[:end])
