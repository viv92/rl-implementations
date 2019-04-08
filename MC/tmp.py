import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

p = np.ones([env.nS, env.nA]) / env.nA
print p
