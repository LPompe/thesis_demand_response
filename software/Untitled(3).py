
# coding: utf-8

# In[1]:

from simulation_enviroment import DemandResponseEviroment
from policies import *
from noise_functions import gaussian_stochastic,identity
from pricing_generators import *
from reward_functions import *
import sys


# In[ ]:




# In[2]:




# In[3]:

def reward_function(s, cells):
    return switch_cost(s, cells) + boundary_exceedence_cost(s, cells)
    return energy_price_cost(s, cells) + switch_cost(s, cells) + boundary_exceedence_cost(s, cells)


# In[4]:

length = 1440
sys.setrecursionlimit(length * 10)


# In[5]:

gamma = -1 * (1/length - 1)
gamma = 0.8
print(gamma)


# In[6]:

env =  DemandResponseEviroment(episode_length=length * 2,  noise_function=gaussian_stochastic, 
                               pricing_generator=ApxShiftPricingGenerator)


# In[20]:

p = QLearningPolicy(alpha=0.3, gamma=gamma, epsilon=0.8)


# In[23]:

rewards = []
def play(p, t, update = True):
    if t > length * 2 + 1:
        return 0
    s , cells = env.get_global_state(), env.cells
    action = p.policy(s, cells)
    env.execute_action(action)
    ns , ncells = env.get_global_state(), env.cells
    reward = reward_function(ns, ncells) + gamma  * play(p, t + 1)
    if t < length:
        if update:
            p.update(s, cells, action, ns, ncells, reward)
        rewards.append(reward)
    return reward


# In[24]:

iters = 100
avg_rewards = []
for i in range(iters):
    if not i % 100:
        print('Starting iter..', i)
    env.start_episode(visualise=False)
    play(p, 0)
    avg_rewards.append(sum(rewards)/len(rewards))
    rewards = []


# In[25]:

f = plt.figure()
f.set_size_inches(16, 9)
_ = plt.plot(avg_rewards)


# In[27]:

rewards = []
env.start_episode(visualise=True)
play(p, 0, update=False)


# In[28]:

print(sum(rewards)/len(rewards))


# In[29]:

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



