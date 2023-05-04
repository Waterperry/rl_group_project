__author__ = 'ny300', 'tb791'
import tensorflow as tf
import numpy as np

from keras.models import clone_model
from keras.optimizers import Adam
from keras.losses import Huber
from UniformReplayBuffer import Experience
from SoftTargetDqn import SoftTargetDqnAgent


# Double DQN Agent
class DoubleDqnAgent(SoftTargetDqnAgent):
    """Double DQN Agent"""

    def __init__(self,
                 obs_shape,  # size of the state |s|
                 num_actions,  # number of actions in the environment
                 alpha=1e-3,  # AdaM learning rate
                 epsilon=0.1,  # random move probability
                 gamma=0.99,  # discount factor
                 qnet_fc_layer_params=(32, 16),  # neuron counts for the fully-connected layers of the Q Network
                 qnet_conv_layer_params=(32, 64, 128),  # filter counts for convolutional layers of the Q network
                 rng_seed: int = None,  # seed to RNG (optional, for debugging really)
                 debug: bool = False,  # enable debugging mode, useful for stack traces in tensorflow functions
                 target_update_period: int = 10,  # how often to update target network (in train steps)
                 epsilon_decay: bool = True,  # decay epsilon whenever batch training takes place
                 decay_factor=0.9999,  # the rate of epsilon decay
                 target_update_tau=0.5,
                 preprocessing_layer=False
                 ):

        super(DoubleDqnAgent, self).__init__(obs_shape=obs_shape, num_actions=num_actions, alpha=alpha,
                                             epsilon=epsilon, gamma=gamma,
                                             qnet_fc_layer_params=qnet_fc_layer_params,
                                             qnet_conv_layer_params=qnet_conv_layer_params,
                                             rng_seed=rng_seed, debug=debug)

        self._cumulative_train_step_counter = 0
        self._target_update_period = target_update_period
        self._target_update_tau = target_update_tau

        # for epsilon decay
        self.decay_factor = decay_factor
        self.epsilon_decay = epsilon_decay

        # Add second target network for action-value estimation.
        self._tnet = clone_model(self._qnet)  # copy the Q-Network structure to target
        self._tnet.compile(optimizer=Adam(learning_rate=alpha), loss=Huber())

    def train_on_batch(self, batch: [Experience]):

        # update target network if enough training steps have passed since last update
        if self._cumulative_train_step_counter >= self._target_update_period:
            self._cumulative_train_step_counter = 0
            self.update_target()
        else:
            self._cumulative_train_step_counter += 1

        # unpack the experience batch into arrays of state, reward, etc.
        states = np.array(list(map(lambda x: x.state, batch)))
        rewards = np.array(list(map(lambda x: x.reward, batch)))
        s_primes = np.array(list(map(lambda x: x.next_state, batch)))

        # convert s_primes to a tensor, determine best actions using qnet,
        # calculate new action values using target network
        t_s_primes = tf.convert_to_tensor(s_primes)
        a_primes = tf.argmax(self._qnet(t_s_primes), axis=1)
        target_q_vals = self._tnet(t_s_primes)

        # calculate action values from target net for actions selected by qnet
        indices = tf.stack([tf.range(len(batch), dtype=tf.int32), tf.cast(a_primes, tf.int32)], axis=1)
        q_prime = tf.gather_nd(target_q_vals, indices)

        # create a mask to apply to the max_q_prime array, because we don't want to consider the max_q value of the
        #   next state if our state s is terminal
        mask = np.array(list(map(lambda x: not x.is_terminal(), batch)), dtype=np.float32)

        y_true = rewards + mask * self._gamma * q_prime

        # finally, convert the above array to a tensor, and train our q_network on it
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        history = self._qnet.fit(x=states, y=y_true, shuffle=False, verbose=False)

        return history.history['loss']
