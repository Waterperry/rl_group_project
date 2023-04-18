import gym
import tensorflow as tf
import numpy as np
from gym.spaces import Discrete
from keras.optimizers.legacy.adam import Adam
from matplotlib import pyplot as plt


class Experience:
    def __init__(self, s, a, r, s_p):
        self.state = s
        self.action = a
        self.reward = r
        self.next_state = s_p

        self._as_list = [s, a, r, s_p]
        self._list_iter_counter = 0

    def __iter__(self):
        self._list_iter_counter += 1
        if self._list_iter_counter < 4:
            yield self._as_list[self._list_iter_counter - 1]
        else:
            raise StopIteration

class DqnAgent:
    def __init__(self,
                 num_observations,  # size of the state |s|
                 num_actions,  # number of actions in the environment
                 alpha=1e-3,   # learning rate
                 epsilon=0.1,  # random move probability
                 gamma=0.99,   # discount factor
                 qnet_layer_params=(128, 64),  # neuron counts for the fully-connected layers of the Q Network
                 rng_seed: int = None  # seed to RNG (optional, for debugging really)
                 ):
        # public fields

        # prefix private (i.e. shouldn't be accessed from other classes) fields with underscore.
        self._action_set = np.array(list(range(num_actions)))
        self._alpha = np.float32(alpha)
        self._epsilon = np.float32(epsilon)
        self._gamma = np.float32(gamma)
        self._rng = np.random.RandomState(seed=rng_seed)

        self._qnet = tf.keras.models.Sequential()
        self._qnet.add(tf.keras.layers.InputLayer(input_shape=(num_observations,)))
        for neuron_count in qnet_layer_params:
            self._qnet.add(tf.keras.layers.Dense(neuron_count, 'relu'))
        self._qnet.add(tf.keras.layers.Dense(num_actions, 'linear'))

        self._qnet.compile(Adam(), loss=self.td_loss, run_eagerly=True)

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

    @staticmethod
    def td_loss(y_true: tf.RaggedTensor, y_pred: tf.Tensor):
        # y_true contains the necessary information to compute TD-Loss.
        # i.e. it's an Experience object and max_q for a_prime
        # assume we only have one sample??
        info = y_true[0]
        a, r, max_q, action_len, gamma, alpha = info

        a = int(a)
        action_len = int(action_len)
        # define loss as a one-hot vector for the action we took.
        loss = tf.one_hot(a, action_len)
        loss *= r + gamma * max_q
        loss -= y_pred
        loss *= alpha

        return loss

    def train(self, experience: Experience):
        s_prime = tf.expand_dims(experience.next_state, axis=0)
        a_prime = self._qnet(s_prime)
        max_q_for_a_prime = tf.squeeze(tf.reduce_max(a_prime)).numpy()
        ins = tf.expand_dims(experience.state, axis=0)
        info = tf.convert_to_tensor((np.float32(experience.action), np.float32(experience.reward),
                                    max_q_for_a_prime, np.float32(len(self._action_set)),
                                    self._gamma, self._alpha))
        info = tf.reshape(info, (1, -1))

        loss = self._qnet.train_on_batch(ins, y=info)
        return loss

    def set_epsilon(self, new_value):
        self._epsilon = new_value


def run_dqn_on_env(env: gym.Env, num_episodes=150):
    assert type(env.action_space) == Discrete, "Can only use this DQN Agent on discrete state spaces for now."

    # reset the environment and get the initial state s0
    state = env.reset()

    # create our DQN agent, passing it information about the environment's observation/action spec.
    # don't worry about any potential warnings in the next line, n must exist if the action space is discrete.
    dqn_agent = DqnAgent(len(state), env.action_space.n)

    # while the episode isn't over, generate a new action on the state, perform that action, then train.
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        ep_return = 0.
        state = env.reset()
        done = False
        while not done:
            # call the action wrapper to get an e-greedy action
            action = dqn_agent.action(state)

            # run the action on the environment and get the new info
            new_state, reward, done, info = env.step(action)

            # render the environment
            env.render('human')

            # train on the experience
            experience = Experience(state, action, reward, new_state)
            loss = dqn_agent.train(experience)
            ep_return += reward

            # print(loss)
            state = new_state

        # episode terminated by this point
        returns[ep] = ep_return

    plt.plot(returns)
    plt.show()
    print(returns)


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
