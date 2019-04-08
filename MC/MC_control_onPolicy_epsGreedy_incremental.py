import gym
import numpy as np
import sys
import matplotlib
from random import randint

if "../" not in sys.path:
    sys.path.append("../")
from collections import defaultdict
from lib.envs.blackjack import BlackjackEnv
from lib import plotting
matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):

    def policy_fn(observation):
        e = randint(1,10)
        if e <= (epsilon*10):
            #take random action
            a = randint(0,nA-1)
        else:
            #take greedy action
            a = np.argmax(Q[observation])
        return a

    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, epsilon, df=1.0):

    #visit counts
    returns_count = defaultdict(float)

    #is visited dictionary for episodic book keeping
    isVisited = defaultdict(int)

    #action-values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    #policy to be learnt
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for ep in range(1, num_episodes+1):
        #print for debug
        if ep % 1000 == 0:
            print "episode no: ", ep, "/", num_episodes
            sys.stdout.flush()
        #generate an episode
        episode = []
        G = 0
        state = env.reset()
        for t in range(100):
            action = policy(state)
            sa_pair = (state, action)
            isVisited[sa_pair] = 0
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            G += (df**t) * reward
            if done:
                break
            state = next_state
        #update action-values for this episode
        for state, action, reward in episode:
            sa_pair = (state, action)
            if isVisited[sa_pair] == 0:
                returns_count[sa_pair] += 1
                #check
                Q[state][action] += (G - Q[state][action]) / returns_count[sa_pair]
                G = (G - reward) / df
                isVisited[sa_pair] = 1

    return Q, policy

Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

# For plotting: Create value function from action-value function# For p
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
