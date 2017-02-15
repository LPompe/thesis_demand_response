
import matplotlib.pyplot as plt
import numpy as np
from random import randint

class BasePricingGenerator():

    def __init__(self, length):
        self.length = length

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
