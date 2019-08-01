
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')

import importlib
import numpy as np
import matplotlib.pyplot as plt
import drone_2d_solver
from drone_2d_solver import Drone_actor, DQNSolver
importlib.reload(drone_2d_solver)


# In[2]:


drone_actor = Drone_actor()
drone_actor.rendering = False


# In[10]:


for i in range(500):
    drone_actor.run_episode(1000)


# In[15]:


drone_actor.dqn_solver.batch_size = len(drone_actor.dqn_solver.memory)
for i in range(1):
    drone_actor.dqn_solver.experience_replay()
    drone_actor.dqn_solver.target_train()

drone_actor.dqn_solver.batch_size = 100
drone_actor.dqn_solver.exploration_rate = 1.0


# In[16]:


x,y = np.meshgrid(range(-25,25), range(-25,25))
x = x.flatten()
y = y.flatten()
distance = np.sqrt(x**2 + y**2)

time = np.zeros(len(x)) + 5
batch = np.stack([x, y, distance, time]).transpose()

q_values = drone_actor.dqn_solver.model.predict(batch)
q_values[q_values == 0] = -200

rewards = q_values.max(axis=1)
reward_im = np.reshape(rewards, (50, 50))
plt.imshow(reward_im)
plt.colorbar()
plt.show()

best_actions = q_values.argmax(axis=1)
best_actions_im = np.reshape(best_actions, (50, 50))
plt.imshow(best_actions_im)
plt.colorbar()
plt.show()


# In[14]:


state = np.array([[25.,25.,30., 7]])
print(drone_actor.dqn_solver.model.predict(state))

state = np.array([[0.,0.,0., 7]])
print(drone_actor.dqn_solver.model.predict(state))

state = np.array([[50.,50.,np.sqrt(25**2 + 25**2), 7]])
print(drone_actor.dqn_solver.model.predict(state))

state = np.array([[21., -12., 24., 1.]])
print(drone_actor.dqn_solver.model.predict(state))

state = np.array([[20.4, -12., 23.7, 1.]])
print(drone_actor.dqn_solver.model.predict(state))


# In[7]:


drone_actor.dqn_solver.memory


# In[35]:


for i in range(200):
    drone_actor.dqn_solver.experience_replay()
    drone_actor.dqn_solver.target_train()


# In[9]:


samples = [[state[0][0], state[0][1], action, reward] for state, action, reward, state_next, terminal in drone_actor.dqn_solver.memory if reward != 0]

samples = np.array(samples)
plt.scatter(samples[:,0], samples[:,1], 4, c=samples[:,3])
plt.colorbar()

