import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.envs.make("Breakout-v0")
print("Action space size: {}".format(env.action_space.n))
#print(env.get_action_meanings())

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

plt.figure()
plt.imshow(env.render(mode='rgb_array'))

for ep in range(100):
    print "ep: ", ep
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(env.action_space.n))
        _, _, done, _ = env.step(action)
        #plt.figure()
        plt.imshow(env.render(mode='rgb_array'))
        if done:
            break

env.render(close=True)
