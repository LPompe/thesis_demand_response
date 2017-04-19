import math
from random import choice
from pricing_generators import BasePricingGenerator
import pandas as pd
from noise_functions import identity
import matplotlib.pyplot as plt
class DemandResponseEviroment():
    """
    Base enviroment to run cooling/heating simulations.

    The demandResponseEviroment will run episodes of a set lenght,
    if the enviroment is used for reinforcement learning:
        we advise an episode length of at least 2 * 1 /( 1 - gamma).
        Where gamma is the discount factor


    The enviroment contains a variable number of cells >= 1 (see respective class)
    if these cells are not defined at init of this demandResponseEviroment object,
    one will be generated.

    The enviroment allows defining of a pricing generator,
    this function will be used to determine the energy pricing at the start of an episode
    if the pricing function remains undefined, energy pricing will be set to zero

    The enviroment allows defining of a noise function,
    this function will add noise to cooling/heating speed of all simulated cells

    The enviroment has a global state, which aggregates all cell states,
    along with total energy usage and pricing for a given timestep

    """

    def __init__(self, episode_length = None, cells = [],
    pricing_generator = BasePricingGenerator, noise_function = identity):

        self.episode_length = episode_length
        self.cells = self.__assign_cells(cells)
        self.episode_in_progress = False
        self.pricing_generator = self.__assign_pricing_generator(pricing_generator)
        self.noise_function = noise_function

        self.episode_history = {}
        self.timestep = 0
        self.episode_pricing = None

        self.visualise = None

    def __assign_cells(self, cells):
        """
        Initializes a cell, or resets cells if provides.
        """
        out_cells = cells
        if cells:
            if not isinstance(cells, list):
                out_cells = [cells]
        else:
            out_cells = [Cell()]

        for cell in out_cells:
            cell.reset()

        return out_cells

    def __assign_pricing_generator(self, pricing_generator):
        if isinstance(pricing_generator, BasePricingGenerator):
            return pricing_generator
        else:
            return pricing_generator(self.episode_length)


    def start_episode(self, visualise = False):

        if self.episode_in_progress and not self.timestep >= self.episode_length:
            self.reset()

        self.reset()
        self.episode_in_progress = True
        self.episode_pricing = self.pricing_generator.generate_episode_pricing()
        self.visualise = visualise

    def reset(self):
        """
        reset all cells to their starting states
        """
        for cell in self.cells:
            cell.reset()

        self.episode_in_progress = False
        self.timestep = 0
        self.episode_pricing = self.pricing_generator.generate_episode_pricing()
        self.episode_history = {}


    def get_global_state(self):
        """
        returns the joined state dictionairies of all cells,
        also includes pricing info and the power usage sum
        """
        global_state_dict = {}
        power_usage = 0
        pricing = self.episode_pricing[self.timestep]


        for cell in self.cells:

            cell_dict = cell.get_state_dict()
            global_state_dict.update(cell_dict)

            if cell.state == Cell.on:
                power_usage += cell.energy_use

        global_state_dict['power_usage'] = power_usage
        global_state_dict['pricing'] = pricing
        if self.timestep < self.episode_length - 60:
            global_state_dict['price_diff'] = pricing - self.episode_pricing[self.timestep + 60]
        else:
            global_state_dict['price_diff'] = 0
        return global_state_dict

    def execute_action(self, a):

        if not self.episode_in_progress or self.timestep >= self.episode_length - 1:
            return

        self.episode_in_progress = True

        if not isinstance(a, list):
            a = [a]

        for cell, action in zip(self.cells, a):
            cell.execute_action(action, noise_function = self.noise_function)

        self.timestep += 1


        self.episode_history[self.timestep] = self.get_global_state()
        if self.visualise:
            if self.timestep >= self.episode_length - 1:
                self.visualise_history()


    def visualise_history(self):
        print('VISUALISING')
        history = self.episode_history
        history_dataframe =  pd.DataFrame(history).T
        f, axes = plt.subplots(2 * len(self.cells) + 3, sharex=True)

        for i, cell in enumerate(self.cells):
            temp_ax = axes[i]
            state_ax = axes[i + 1]

            temp_data = history_dataframe[cell.name + '_temp']
            max_data = history_dataframe[cell.name + '_max']
            min_data = history_dataframe[cell.name + '_min']

            state_data = history_dataframe[cell.name + '_state']

            temp_ax.plot(temp_data, label = cell.name + '_temp')
            temp_ax.plot(max_data, label = cell.name + '_max')
            temp_ax.plot(min_data, label = cell.name + '_min')
            temp_ax.legend()

            state_ax.plot(state_data, label = cell.name + '_state')
            state_ax.legend()

        axes[-3].plot(history_dataframe['power_usage'], label = 'power_consumption')
        axes[-3].legend()
        axes[-2].plot(history_dataframe['pricing'], label = 'pricing')
        axes[-2].legend()
        realised_price  = history_dataframe['pricing'] * history_dataframe['power_usage']
        axes[-1].plot(realised_price, label = 'realised_price')
        axes[-1].legend()

        print('realised_price:', realised_price.sum())
        f.set_size_inches(16, 9)
        plt.show()






class Cell():
    counter = 0 # for cell naming purposes
    on = 1
    off = 0

    def __init__(self, cooling_speed = 0.1, heating_speed = 0.1,
    min_temp = 0, max_temp = 1, energy_use = 0.5):

        self.name = 'cell_' + str(Cell.counter)

        self.start_state = self.set_start_state()
        self.state = self.start_state

        self.max_temp = max_temp
        self.min_temp = min_temp

        self.starting_temperature = abs(max_temp - min_temp)/2 + min_temp
        self.temperature = self.starting_temperature

        self.cooling_speed = cooling_speed
        self.heating_speed = heating_speed
        self.time_on = 0
        self.time_off = 0

        self.energy_use = energy_use

        Cell.counter += 1

    def set_start_state(self):
        self.start_state = choice([Cell.on, Cell.off])
        return self.start_state

    def reset(self):
        self.state = self.start_state
        self.temperature = self.starting_temperature

    def get_state_dict(self):
        return {
        self.name + '_temp' : self.temperature,
        self.name + '_max' : self.max_temp,
        self.name + '_min' : self.min_temp,
        self.name + '_state' : self.state,
        }

    def execute_action(self, a, noise_function = identity):
        self.state = a
        if a == Cell.on:
            self.temperature -= noise_function(((self.temperature + 0.1) * math.log(1.022, math.exp(1))))
            self.time_off = 0
            self.time_on += 1

        elif a == Cell.off:
            self.temperature += noise_function((1 / (self.temperature + 0.1) * math.log(1.00185, math.exp(1))))
            self.time_on = 0
            self.time_off += 1
