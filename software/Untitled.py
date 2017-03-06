
# coding: utf-8

# In[3]:

from simulation_enviroment import DemandResponseEviroment
from policies import *
from noise_functions import gaussian_stochastic,identity
from pricing_generators import *
from reward_functions import *
import sys


# In[ ]:




# In[4]:



# In[5]:

def reward_function(s, cells):
    #return boundary_exceedence_cost(s, cells)
    return energy_price_cost(s, cells) + switch_cost(s, cells) + boundary_exceedence_cost(s, cells)


# In[6]:

length = 1440
sys.setrecursionlimit(length * 10)


# In[7]:

gamma = -1 * (1/length - 1)
gamma = 0.8
print(gamma)


# In[8]:

env =  DemandResponseEviroment(episode_length=length * 2,  noise_function=identity, 
                               pricing_generator=ApxShiftPricingGenerator)


# In[9]:

p = QLearningRfPolicy(alpha=0.4, gamma=gamma, epsilon=0.99, length = length)


# In[10]:

rewards = []
def play(p, t, update = True):
    if t > length * 2 + 1:
        return 0
    s , cells = env.get_global_state(), env.cells
    action = p.policy(s, cells)
    env.execute_action(action)
    ns , ncells = env.get_global_state(), env.cells
    reward = reward_function(ns, ncells) + gamma  * play(p, t + 1, update)
    if t < length:
        if update:
            p.update(s, cells, action, ns, ncells, reward)
        rewards.append(reward)
    return reward


# In[11]:

iters = 150
avg_rewards = []
for i in range(iters):
    env.start_episode(visualise=False)
    r = play(p, 0)
    avg = sum(rewards)/len(rewards)
    avg_rewards.append(avg)
    rewards = []
    if not i % 10:
        print('Starting iter..', i, avg)


# In[ ]:

f = plt.figure()
f.set_size_inches(16, 9)
_ = plt.plot(avg_rewards)


# In[ ]:

rewards = []
env.start_episode(visualise=True)
play(p, 0, update=False)
print(sum(rewards)/len(rewards))


# In[ ]:

qp = LatestSwitchPolicy(reward_function)
rewards = []
env.start_episode(visualise=True)
play(qp, 0, update=False)
print(sum(rewards)/len(rewards))


# In[ ]:

p.qvalues


# In[ ]:

p.epsilon


# In[ ]:



