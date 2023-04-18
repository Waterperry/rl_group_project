import tensorflow as tf
import numpy as np


class DqnAgent:
    def __init__(self,
                 action_set=(),                 # a tuple/list of all the possible actions in the environment
                 epsilon=0.1,                   # random move probability
                 qnet_layer_params=(128, 64),   # neuron counts for the fully-connected layers of the Q Network
                 rng_seed: int = None           # seed to RNG (optional, for debugging really)
                 ):
        # public fields

        # prefix private (i.e. shouldn't be accessed from other classes) fields with underscore.
        self._action_set = action_set
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed=rng_seed)

        self._qnet = tf.keras.models.Sequential()
        for neuron_count in qnet_layer_params:
            self._qnet.add(tf.keras.layers.Dense(neuron_count))

    def action(self, state):
        # rolled less than epsilon. return random action.
        if self._rng.random() < self._epsilon:
            return self._rng.choice(self._action_set)

        # choose a greedy action
        # generate action q values by calling the network on the current state.
        # qnet may expect a batched input, in which case we need to expand dims.
        action_q_values = self._qnet(tf.expand_dims(state, axis=0))
        action_q_values = tf.squeeze(action_q_values)

        return tf.squeeze(tf.argmax(action_q_values))

    def train(self, experience_batch):
        pass

    def set_epsilon(self, new_value):
        self._epsilon = new_value


if __name__ == '__main__':
    example_state = (0., 0., 1.)
    example_action_set = list(range(5))
    dude = DqnAgent(action_set=example_action_set, epsilon=1., rng_seed=150)
    print(f"Sampling a random action: {dude.action(example_state)}")

    dude.set_epsilon(0.)
    print(f"Sampling a greedy action: {dude.action(example_state)}")
