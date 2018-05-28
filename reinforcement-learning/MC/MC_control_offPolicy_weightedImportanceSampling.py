
import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def create_random_policy(nA):

    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn


def create_greedy_policy(Q, nA):

    def policy_fn(observation):
        A = np.eye(nA)[np.argmax(Q[observation])]
        # A = np.zeros_like(Q[observation], dtype=float)
        # best_action = np.argmax(Q[observation])
        # A[best_action] = 1.0
        return A
    return policy_fn


def mc_control_importance_sampling(env, num_episodes, behavior_policy, df=1.0):

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    #cumulated denominator of importance sampling formula
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q, env.action_space.n)

    for ep in range(1, num_episodes+1):
        #print for debug
        if ep % 1000 == 0:
            print "episode no: ", ep, "/", num_episodes
            sys.stdout.flush()
        #generate an episode
        episode = []
        state = env.reset()
        for t in range(100):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(env.action_space.n), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        #update action-values for this episode
        W = 1.0 #importance sampling ratio
        G = 0.0 #return for the episode
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = (df*G) + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action != np.argmax(target_policy(state)):
                break
            W *= 1./(behavior_policy(state)[action])

    return Q, target_policy

random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)


# For plotting: Create value function from action-value function# For p
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
