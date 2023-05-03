__author__ = 'ny300'

from typing import List
import tensorflow as tf
import numpy as np
from keras.models import clone_model

from DqnAgent import DqnAgent
from keras.losses import Huber
from keras.optimizers.legacy.adam import Adam

#Experience, but with added varable for n_step_rewards
class Experience:
    
    def __init__(self, s, done, a, reward, s_p, n_step_rewards=None):
        if n_step_rewards is None:
            n_step_rewards = []
        self.state = s
        self.done = done
        self.action = a
        self.n_step_rewards = n_step_rewards
        self.reward = reward
        self.next_state = s_p

        self._as_list = [s, a, reward, s_p]
        self._list_iter_counter = 0

    def __iter__(self):
        self._list_iter_counter += 1
        if self._list_iter_counter < 4:
            yield self._as_list[self._list_iter_counter - 1]
        else:
            raise StopIteration

    def is_terminal(self):
        return self.done
    
    
# Double DQN Agent
class DoubleDqnAgent(DqnAgent):
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
                 update_interval: int = 10,  # how often the target network is updated with the weights from the qnet
                 epsilon_decay: bool = True,  # decay epsilon whenever batch training takes place
                 decay_factor=0.9999,  # the rate of epsilon decay
                 target_update_tau=0.5,
                 preprocessing_layer=False
                 ):
        
        super(DoubleDqnAgent, self).__init__(obs_shape=obs_shape, num_actions=num_actions, alpha=alpha,
                                             epsilon=epsilon,gamma=gamma,
                                             qnet_fc_layer_params=qnet_fc_layer_params,
                                             qnet_conv_layer_params=qnet_conv_layer_params,
                                             preprocessing_layer=preprocessing_layer,
                                             rng_seed=rng_seed, debug=debug)
        
        self._train_count = 0
        self.update_interval = update_interval
        self._target_update_tau = target_update_tau
        
        # for epsilon decay
        self.decay_factor = decay_factor
        self.epsilon_decay = epsilon_decay
        
        # Add second target network for action-value estimation.
        self._tnet = clone_model(self._qnet)  # copy the Q-Network structure to target
        self._tnet.compile(optimizer=Adam(learning_rate=alpha), loss=Huber())
    
    def train_on_batch(self, batch: [Experience]):
        
        # update target network if enough training steps have passed since last update
        if self._train_count >= self.update_interval:
            self._train_count = 0 
            self.update_target_weights()
        else: 
            self._train_count += 1
            
        # unpack the experience batch into arrays of state, reward, etc.
        states = np.array(list(map(lambda x: x.state, batch)))
        rewards = np.array(list(map(lambda x: x.reward, batch)))
        s_primes = np.array(list(map(lambda x: x.next_state, batch)))

        # convert s_primes to a tensor, determine best actions using qnet,
        # calculate new action values using target network
        t_s_primes = tf.convert_to_tensor(s_primes)
        actions = tf.argmax(self._qnet(t_s_primes), axis=1)
        target_q_vals = self._tnet(t_s_primes)
        
        # calculate action values from target net for actions selected by qnet 
        indices = tf.stack([tf.range(len(batch), dtype=tf.int32), tf.cast(actions, tf.int32)], axis=1)
        q_prime = tf.gather_nd(target_q_vals, indices)

        # create a mask to apply to the max_q_prime array, because we don't want to consider the max_q value of the
        #   next state if our state s is terminal
        mask = np.array(list(map(lambda x: x.is_terminal(), batch)), dtype=np.float32)
        
        y_true = rewards + mask * self._gamma * q_prime

        # finally, convert the above array to a tensor, and train our q_network on it
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        history = self._qnet.fit(x=states, y=y_true, shuffle=False, verbose=False)
        
        if self.epsilon_decay:
            self.do_epsilon_decay()
        
        return history.history['loss']
    
    # copying the weights from the qnet onto the target
    def update_target_weights(self):
        qnet_weights = self._qnet.get_weights()
        target_weights = self._tnet.get_weights()

        # Find difference between qnet_weights and target_weights, times by tau (0.5) and add to target_weights
        modified_qnet_weights = [self._target_update_tau * i for i in qnet_weights]
        modified_target_weights = [(1 - self._target_update_tau) * j for j in target_weights]
        weights = [modified_target_weights[x] + modified_qnet_weights[x] for x in range(len(modified_target_weights))]
        self._tnet.set_weights(weights)  # copy the Q-Network weights to target
        self._train_count = 0

        
    # epsilon decay 
    def do_epsilon_decay(self):
        self.set_epsilon(self._epsilon * self.decay_factor)


# N-step Ddqn agent
class NStepDdqnAgent(DoubleDqnAgent):
        
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
                 update_interval: int = 10,  # how often the target network is updated with the weights from the qnet
                 epsilon_decay: bool = True,  # decay epsilon whenever batch_training takes place
                 decay_factor=0.9999,  # the rate of epsilon decay
                 preprocessing_layer=False,
                 n_steps=5,  # n-step targets
                 target_update_tau=0.5,
                 ):
        
        super(NStepDdqnAgent, self).__init__(obs_shape=obs_shape, num_actions=num_actions, alpha=alpha,epsilon=epsilon,
                                             gamma=gamma, qnet_fc_layer_params=qnet_fc_layer_params,
                                             qnet_conv_layer_params=qnet_conv_layer_params, rng_seed=rng_seed,
                                             debug=debug, update_interval=update_interval, epsilon_decay=epsilon_decay,
                                             decay_factor=decay_factor, preprocessing_layer=preprocessing_layer,
                                             target_update_tau=target_update_tau)
        
        self.n_steps = n_steps

    def train_on_batch(self, batch: List[Experience]):
        
        # update target network if enough training steps have passed since last update
        if self._train_count >= self.update_interval:
            self._train_count = 0 
            self.update_target_weights()
        else: 
            self._train_count += 1
        
        # unpack the experience batch into arrays of state, reward, etc.
        states = np.array(list(map(lambda x: x.state, batch)))
        rewards_sequence = np.array(list(map(lambda x: x.n_step_rewards, batch)))
        s_primes = np.array(list(map(lambda x: x.next_state, batch)))

        # convert s_primes to a tensor, determine best actions using qnet,
        # calculate new action values using target network
        t_s_primes = tf.convert_to_tensor(s_primes)
        qnet_outs = []
        for s_prime_tensor in t_s_primes:
            qnet_outs.append(tf.squeeze(self._qnet(tf.expand_dims(s_prime_tensor, axis=0))))

        qnet_outs = tf.convert_to_tensor(qnet_outs)
        actions = tf.argmax(qnet_outs, axis=1)
        target_q_vals = self._tnet(t_s_primes)
        
        # calculate action values from target net for actions selected by qnet
        range = tf.range(len(batch), dtype=tf.int32)
        actions_conv = tf.cast(actions, tf.int32)
        indices = tf.stack((range, actions_conv), axis=1)
        q_prime = tf.gather_nd(target_q_vals, indices)

        # create a mask to apply to the max_q_prime array, because we don't want to consider the max_q value of the
        #   next state if our state s is terminal
        mask = np.array(list(map(lambda x: x.is_terminal(), batch)), dtype=np.float32)
        
        # Calculate decayed sum of n-step rewards
        n_step_rewards = np.array([sum((self._gamma**i) * r for i, r in enumerate(rewards))
                                                            for rewards in rewards_sequence])
        
        y_true = n_step_rewards + mask * self._gamma * q_prime

        # finally, convert the above array to a tensor, and train our q_network on it
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        # hack batch_size=1, so we can get loss per sample (for PER)
        history = self._qnet.fit(x=states, y=y_true, epochs=1, shuffle=False, verbose=False)
        
        if self.epsilon_decay:
            self.do_epsilon_decay()
        
        return history.history['loss']
