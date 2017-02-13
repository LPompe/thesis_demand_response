
import matplotlib.pyplot as plt
import numpy as np
from random import randint

def identity(v):
    return v

def gaussian_stochastic(v):
    sigma = 0.05
    return v + np.random.normal(0, sigma)
