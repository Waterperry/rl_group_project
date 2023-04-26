import os
import tensorflow as tf
import numpy as np
from keras.losses import Huber
from keras.optimizers.legacy.adam import Adam
from UniformReplayBuffer import Experience


class DqnAgent:
    def __init__(self,
                 obs_shape,  # size of the state |s|
                 num_actions,  # number of actions in the environment
                 alpha=1e-3,  # AdaM learning rate
                 epsilon=0.1,  # random move probability
                 gamma=0.99,  # discount factor
                 qnet_fc_layer_params=(128, 64),  # neuron counts for the fully-connected layers of the Q Network
                 qnet_conv_layer_params=(32, 64, 128),  # filter counts for convolutional layers of the Q network
                 rng_seed: int = None,  # seed to RNG (optional, for debugging really)
                 debug: bool = False  # enable debugging mode, useful for stack traces in tensorflow functions
                 ):
        # public fields

        # prefix private (i.e. shouldn't be accessed from other classes) fields with underscore.
        self._action_set = np.array(list(range(num_actions)))
        self._alpha = np.float32(alpha)
        self._epsilon = np.float32(epsilon)
        self._gamma = np.float32(gamma)
        self._rng = np.random.RandomState(seed=rng_seed)
        self._debug = debug

        self._qnet = tf.keras.models.Sequential()
        self._qnet.add(tf.keras.layers.InputLayer(input_shape=obs_shape))

        # if we need to use convolutional layers, add them and pooling layers after.
        if qnet_conv_layer_params is not None:
            for filter_count in qnet_conv_layer_params:
                self._qnet.add(tf.keras.layers.Conv2D(filter_count, kernel_size=3, activation='relu'))
                self._qnet.add(tf.keras.layers.MaxPool2D())
                self._qnet.add(tf.keras.layers.BatchNormalization())

            self._qnet.add(tf.keras.layers.Flatten())

        # add the fully-connected layers to the neural network.
        for neuron_count in qnet_fc_layer_params:
            self._qnet.add(tf.keras.layers.Dense(neuron_count, 'relu'))

        # add the final layer, with linear activation
        self._qnet.add(tf.keras.layers.Dense(num_actions, 'linear'))

        self._qnet.compile(Adam(learning_rate=alpha), loss=Huber(), run_eagerly=self._debug)

    def action(self, state):
        # rolled less than epsilon. return random action.
        if self._rng.random() < self._epsilon:
            return self._rng.choice(self._action_set)

        # choose a greedy action
        # generate action q values by calling the network on the current state.
        # qnet may expect a batched input, in which case we need to expand dims.
        action_q_values = self._qnet(tf.expand_dims(state, axis=0))
        action_q_values = tf.squeeze(action_q_values)

        return tf.argmax(action_q_values).numpy()

    def train(self, experience: Experience):
        return self.train_on_batch([experience])
        # s_prime = tf.expand_dims(experience.next_state, axis=0)
        # a_prime = self._qnet(s_prime)
        # max_q_for_a_prime = tf.squeeze(tf.reduce_max(a_prime)).numpy()
        # ins = tf.expand_dims(experience.state, axis=0)
        # info = tf.convert_to_tensor((np.float32(experience.action), np.float32(experience.reward),
        #                             max_q_for_a_prime, np.float32(len(self._action_set)),
        #                             self._gamma, self._alpha))
        # info = tf.reshape(info, (1, -1))
        #
        # loss = self._qnet.train_on_batch(ins, y=info)
        # return loss

    def train_on_batch(self, batch: [Experience]):
        # unpack the experience batch into arrays of state, reward, etc.
        states = np.array(list(map(lambda x: x.state, batch)))
        rewards = np.array(list(map(lambda x: x.reward, batch)))
        s_primes = np.array(list(map(lambda x: x.next_state, batch)))

        # convert s_primes to a tensor, run it through our q_network, and get the maximum action value for each s'
        a_primes = self._qnet(tf.convert_to_tensor(s_primes))
        max_q_prime = np.max(a_primes, axis=1)

        # create a mask to apply to the max_q_prime array, because we don't want to consider the max_q value of the
        #   next state if our state s is terminal
        mask = np.array(list(map(lambda x: not x.is_terminal(), batch)), dtype=np.float32)
        y_true = rewards + mask * self._gamma * max_q_prime

        # finally, convert the above array to a tensor, and train our q_network on it
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        history = self._qnet.fit(x=states, y=y_true, shuffle=False, verbose=False)
        return history.history['loss']

    def set_epsilon(self, new_value) -> None:
        # just good practice to have a setter rather than accessing it raw, because we are using conversions as
        #   type checking is pretty important if we are going to try and run our code in TF graph mode.
        self._epsilon = np.float32(new_value)

    def save_policy(self):
        nonce = np.random.randint(0, 1e6)
        if not os.path.isdir('saved_configs'):
            os.makedirs('saved_configs')
        conf = open(f"./saved_configs/dqn_conf_{nonce}.txt", "w+")
        params = vars(self)
        for param in params:
            conf.write(f"{param}: {params[param]}\n")
        self._qnet.save(f"./saved_configs/dqn_qnet_{nonce}.h5")
