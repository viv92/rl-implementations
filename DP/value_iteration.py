import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, v, df=1.0, theta=0.00001):
    v_new = np.zeros(env.nS)
    for i in range(env.nS):
        for j in range(env.nA):
            for prob, next_state, reward, done in env.P[i][j]:
                v_new[i] += (policy[i][j] * (reward + (df * prob * v[next_state])))
    for i in range(env.nS):
        v[i] = v_new[i]
    return np.array(v)

def policy_improvement(policy, env, vf, df=1.0):
    for i in range(env.nS):
        max_val = -1 * float("inf")
        best_a = -1
        for j in range(env.nA):
            for prob, next_state, reward, done in env.P[i][j]:
                val = reward + (df * prob * vf[next_state])
                if max_val < val:
                    max_val = val
                    best_a = j
        for j in range(env.nA):
            policy[i] = np.eye(env.nA)[best_a]
    return np.array(policy)

#initiate with uniform policy and zeroed value function
p = np.ones([env.nS, env.nA]) / env.nA
v = np.zeros(env.nS)

#policy convergence flag
policy_stable = False

#policy iteration loop
while (not policy_stable):
    p_backup = np.copy(p)
    v = policy_eval(p, env, v)
    p = policy_improvement(p, env, v)
    for i in range(env.nS):
        if np.argmax(p_backup[i]) != np.argmax(p[i]):
            break
    if i == (env.nS - 1):
        policy_stable = True

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
print "expected_v: ", expected_v
print "evaluated_v: ", v
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
