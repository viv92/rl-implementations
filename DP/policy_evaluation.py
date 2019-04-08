import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, v, df=1.0, theta=0.00001):
    while True:
        v_new = np.zeros(env.nS)
        delta = 0
        for i in range(env.nS):
            for j in range(env.nA):
                for prob, next_state, reward, done in env.P[i][j]:
                    v_new[i] += (policy[i][j] * (reward + (df * prob * v[next_state])))
            delta = max(abs(v_new[i] - v[i]), delta)
        if delta > theta:
            for i in range(env.nS):
                v[i] = v_new[i]
        else:
            break
    return np.array(v)

#initiate with uniform policy and zeroed value function
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = np.zeros(env.nS)
v = policy_eval(random_policy, env, v)

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
print "expected_v: ", expected_v
print "evaluated_v: ", v
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
