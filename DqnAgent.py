import gym
import tensorflow as tf
import numpy as np
from gym.spaces import Discrete


class DqnAgent:
    def __init__(self,
                 num_observations,  # size of the state |s|
                 num_actions,  # number of actions in the environment
                 epsilon=0.1,  # random move probability
                 qnet_layer_params=(128, 64),  # neuron counts for the fully-connected layers of the Q Network
                 rng_seed: int = None  # seed to RNG (optional, for debugging really)
                 ):
        # public fields

        # prefix private (i.e. shouldn't be accessed from other classes) fields with underscore.
        self._action_set = np.array(list(range(num_actions)))
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed=rng_seed)

        self._qnet = tf.keras.models.Sequential()
        self._qnet.add(tf.keras.layers.InputLayer(input_shape=(num_observations,)))
        for neuron_count in qnet_layer_params:
            self._qnet.add(tf.keras.layers.Dense(neuron_count, 'relu'))
        self._qnet.add(tf.keras.layers.Dense(num_actions, 'linear'))

    def action(self, state):
        # rolled less than epsilon. return random action.
        if self._rng.random() < self._epsilon:
            return self._rng.choice(self._action_set)

        # choose a greedy action
        # generate action q values by calling the network on the current state.
        # qnet may expect a batched input, in which case we need to expand dims.
        action_q_values = self._qnet(tf.expand_dims(state, axis=0))
        action_q_values = tf.squeeze(action_q_values)

        return tf.squeeze(tf.argmax(action_q_values)).numpy()

    def train(self, experience):
        pass

    def set_epsilon(self, new_value):
        self._epsilon = new_value


def run_dqn_on_env(env: gym.Env):
    assert type(env.action_space) == Discrete, "Can only use this DQN Agent on discrete state spaces for now."

    # reset the environment and get the initial state s0
    state = env.reset()

    # create our DQN agent, passing it information about the environment's observation/action spec.
    # don't worry about any potential warnings in the next line, n must exist if the action space is discrete.
    dqn_agent = DqnAgent(len(state), env.action_space.n)

    # while the episode isn't over, generate a new action on the state, perform that action, then train.
    done = False
    while not done:
        # call the action wrapper to get an e-greedy action
        action = dqn_agent.action(state)

        # run the action on the environment and get the new info
        state, reward, done, info = env.step(action)

        # render the environment
        env.render('human')


if __name__ == '__main__':
    # a really simple example observation and action space. Importantly, num_observations is the size of a state
    #    and num_actions is the number of actions available in the environment.
    example_state = (0., 0., 1.)
    example_action_set = list(range(5))

    dude = DqnAgent(num_observations=len(example_state), num_actions=len(example_action_set), epsilon=1., rng_seed=150)
    print(f"Sampling a random action: {dude.action(example_state)}")

    dude.set_epsilon(0.)
    print(f"Sampling a greedy action: {dude.action(example_state)}")

    test_env = gym.make('CartPole-v1')
    run_dqn_on_env(test_env)
