import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

def  make_epsilon_greedy_policy (Q, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def make_greedy_policy (Q, nA):

    def policy_fn(observation):
        A = np.zeros(nA)
        best_action = np.argmax(Q[observation])
        A[best_action] = 1
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy_mu = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    # The policy we are improving
    policy_pi = make_greedy_policy(Q, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1 == 0:
            print "\rEpisode: ", i_episode + 1, " / ", num_episodes
            sys.stdout.flush()

        # Implement this!
        state = env.reset()
        while True:
            action_probs = policy_mu(state)
            action = np.random.choice(np.arange(env.action_space.n), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            pi_next_action_probs = policy_pi(next_state)
            pi_next_action = np.argmax(pi_next_action_probs)
            Q[state][action] += alpha * ((reward + discount_factor * Q[next_state][pi_next_action]) - Q[state][action])
            state = next_state
            stats.episode_lengths[i_episode] += 1
            stats.episode_rewards[i_episode] += reward
            if done:
                break

    return Q, stats

Q, stats = q_learning(env, 500)
plotting.plot_episode_stats(stats)
