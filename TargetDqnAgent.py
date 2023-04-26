from copy import deepcopy

from keras.losses import Huber
from keras.optimizers import Adam

from DqnAgent import DqnAgent
from UniformReplayBuffer import Experience
import numpy as np
from tensorflow import convert_to_tensor, float32 as tf_float32
from keras.models import clone_model


class TargetDqnAgent(DqnAgent):
    def __init__(self,
                 obs_shape,  # size of the state |s|
                 num_actions,  # number of actions in the environment
                 target_update_period=25,   # number of train steps to update target network after

                 alpha=1e-3,  # AdaM learning rate
                 epsilon=0.1,  # random move probability
                 gamma=0.99,  # discount factor
                 qnet_fc_layer_params=(128, 64),  # neuron counts for the fully-connected layers of the Q Network
                 qnet_conv_layer_params=(32, 64, 128),  # filter counts for convolutional layers of the Q network
                 rng_seed: int = None,  # seed to RNG (optional, for debugging really)
                 debug: bool = False  # enable debugging mode, useful for stack traces in tensorflow functions
                 ):

        super().__init__(obs_shape, num_actions, alpha=alpha, epsilon=epsilon, gamma=gamma,
                         qnet_fc_layer_params=qnet_fc_layer_params, qnet_conv_layer_params=qnet_conv_layer_params,
                         rng_seed=rng_seed, debug=debug)

        # target network stuff
        self._target_update_counter = 0     # counter of how many train steps we've done since target update
        self._target_update_period = target_update_period   # the period after which we update target
        self._tnet = clone_model(self._qnet)    # copy the Q-Network structure to target
        self._tnet.compile(optimizer=Adam(learning_rate=alpha), loss=Huber())

    def update_target(self):
        weights = self._qnet.get_weights()
        self._tnet.set_weights(weights)  # copy the Q-Network weights to target
        self._target_update_counter = 0

    def train_on_batch(self, batch: [Experience]):
        # check if we need to update the target network
        if self._target_update_period <= self._target_update_counter:
            self.update_target()

        # unpack the experience batch into arrays of state, reward, etc.
        states = np.array(list(map(lambda x: x.state, batch)))
        rewards = np.array(list(map(lambda x: x.reward, batch)))
        s_primes = np.array(list(map(lambda x: x.next_state, batch)))

        # convert s_primes to a tensor, run it through our q_network, and get the maximum action value for each s'
        a_primes = self._tnet(convert_to_tensor(s_primes))
        max_q_prime = np.max(a_primes, axis=1)

        # create a mask to apply to the max_q_prime array, because we don't want to consider the max_q value of the
        #   next state if our state s is terminal
        mask = np.array(list(map(lambda x: not x.is_terminal(), batch)), dtype=np.float32)
        y_true = rewards + mask * self._gamma * max_q_prime

        # finally, convert the above array to a tensor, and train our q_network on it
        y_true = convert_to_tensor(y_true, dtype=tf_float32)
        history = self._qnet.fit(x=states, y=y_true, shuffle=False, verbose=False)

        # update the target_network counter
        self._target_update_counter += 1
        return history.history['loss']
