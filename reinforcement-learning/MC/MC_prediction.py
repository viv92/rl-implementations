import gym
import numpy as np
import sys
import matplotlib

if "../" not in sys.path:
    sys.path.append("../")
from collections import defaultdict
from lib.envs.blackjack import BlackjackEnv
from lib import plotting
matplotlib.style.use('ggplot')

env = BlackjackEnv()

def mc_prediction(policy, env, num_ep, df=1.0):
    #returns and visit counts
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    #is visited dictionary for episodic book keeping
    isVisited = defaultdict(int)

    #value function
    V = defaultdict(float)

    for ep in range(1, num_ep+1):
        #print for debug
        if ep % 1000 == 0:
            print "episode no: ", ep, "/", num_ep
            sys.stdout.flush()
        #generate an episode
        episode = []
        G = 0
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            isVisited[state] = 0
            G += (df**t) * reward
            if done:
                break
            state = next_state
        #update value function for this episode
        for state, _, reward in episode:
            if isVisited[state] == 0:
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]
                G = (G - reward) / df
                isVisited[state] = 1

    return V

#fixed policy for which value function is to be evaluated
def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


V_10k  =  mc_prediction(sample_policy, env, 10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, 500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
