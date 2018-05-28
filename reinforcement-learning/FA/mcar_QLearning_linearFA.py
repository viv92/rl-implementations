import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

env = gym.envs.make("MountainCar-v0")

# Feature Preprocessing: Normalize to zero mean and unit variance# Featu
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurized representation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        # TODO: Implement this!
        feature_vector = self.featurize_state(s)
        #feature_vector = feature_vector.reshape(-1,1)
        #print "state size: ", s.shape
        #print "feature_vector size: ", feature_vector.shape
        predictions = []
        if a == None:
            for i in range(env.action_space.n):
                prediction = self.models[i].predict(feature_vector)
                #print "prediction size: ", prediction.shape
                predictions.append(prediction)
        else:
            prediction = self.models[a].predict(feature_vector)
            #print "prediction: ", prediction
        return predictions if (a == None) else prediction

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        # TODO: Implement this!
        feature_vector = self.featurize_state(s)
        self.models[a].partial_fit(feature_vector, y)
        return None

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def make_greedy_policy(estimator, nA):

    def policy_fn(observation):
        A = np.zeros(nA)
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] = 1
        return A
    return policy_fn

def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    plt.figure()

    for i_episode in range(num_episodes):

        # The policy we're following
        policy_mu = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        # The policy we're following
        policy_pi = make_greedy_policy(
            estimator, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print "\rEpisode: ", i_episode + 1, " / ", num_episodes, " (", last_reward, ")"
        sys.stdout.flush()

        # TODO: Implement this!
        state = env.reset()
        for t in itertools.count():
            action_probs = policy_mu(state)
            action = np.random.choice(np.arange(env.action_space.n), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            if i_episode > 0:
                plt.imshow(env.render(mode='rgb_array'))
            pi_next_action_probs = policy_pi(state)
            pi_next_action = np.argmax(pi_next_action_probs)
            #print "next_state:", next_state, " pi_next_action:", pi_next_action
            estimate = estimator.predict(next_state, pi_next_action)
            #print "estimate: ", estimate
            target = reward + (discount_factor * estimate)
            estimator.update(state, action, target)
            state = next_state
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward
            if done:
                break

    env.render(close=True)
    return stats

estimator = Estimator()


# Note: For the Mountain Car we don't actually need an epsilon > 0.0
# because our initial estimate for all states is too "optimistic" which leads
# to the exploration of all states.
stats = q_learning(env, estimator, 600, epsilon=0.999, epsilon_decay=0.998)

plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)
