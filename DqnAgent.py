import os

import gym
import tensorflow as tf
import numpy as np
from gym.spaces import Discrete
from keras.losses import Huber
from keras.optimizers.legacy.adam import Adam
from matplotlib import pyplot as plt
from UniformReplayBuffer import UniformReplayBuffer, Experience

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


def cartpole_test(num_episodes=150, render=True, verbose=False):
    env = gym.make('CartPole-v1', render_mode='human')

    # reset the environment and get the initial state s0
    state = env.reset()
    if len(state) == 2 and type(state) == tuple:
        state = state[0]

    # create our DQN agent, passing it information about the environment's observation/action spec.
    dqn_agent = DqnAgent(state.shape, env.action_space.n,
                         qnet_conv_layer_params=None, epsilon=1e-3, alpha=1e-7, gamma=0.9)

    replay_buffer = UniformReplayBuffer(max_length=10000, minibatch_size=128)

    if render:
        print("[WARN]: Rendering will slow down training. Are you sure you want to be rendering?")

    # while the episode isn't over, generate a new action on the state, perform that action, then train.
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        ep_return = 0.
        state = env.reset()
        done = False
        while not done:
            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                state = state[0]
                # call the action wrapper to get an e-greedy action
            action = dqn_agent.action(state)

            # run the action on the environment and get the new info
            new_state, reward, done, truncated, info = env.step(action)

            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                new_state = new_state[0]
            # add the experience to our replay buffer
            experience = Experience(state, done, action, reward, new_state)
            replay_buffer.add_experience(experience)

            # render the environment
            if render:
                env.render()

            # train on the experience
            if not done:
                if replay_buffer.mb_size < replay_buffer.num_experiences():
                    training_batch = replay_buffer.sample_minibatch()
                    loss = dqn_agent.train_on_batch(training_batch)

                    if verbose:
                        print(loss)

            ep_return += reward

            state = new_state

        # episode terminated by this point
        returns[ep] = ep_return
        print(f"Episode {ep} over. Total return: {ep_return}")

    plt.plot(returns)
    _ = plt.title("Agent total returns per episode (Training)"), plt.xlabel("Episode"), plt.ylabel("Return")
    plt.show()
    return dqn_agent, returns


def run_dqn_on_env(env: gym.Env, num_episodes=150, render=True, verbose=False):
    assert type(env.action_space) == Discrete, "Can only use this DQN Agent on discrete action spaces for now."

    # reset the environment and get the initial state s0
    state = env.reset()
    if len(state) == 2 and type(state) == tuple:
        state = state[0]
    # create our DQN agent, passing it information about the environment's observation/action spec.
    dqn_agent = DqnAgent(state.shape, env.action_space.n, qnet_fc_layer_params=(256, 256, 128), epsilon=0.2, gamma=0.95)

    replay_buffer = UniformReplayBuffer(max_length=10000, minibatch_size=32)

    if render:
        print("[WARN]: Rendering will slow down training. Are you sure you want to be rendering?")

    # while the episode isn't over, generate a new action on the state, perform that action, then train.
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        ep_return = 0.
        state = env.reset()

        done = False
        while not done:
            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                state = state[0]
                # call the action wrapper to get an e-greedy action
            action = dqn_agent.action(state)

            # run the action on the environment and get the new info
            new_state, reward, truncated, done, info = env.step(action)

            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                new_state = new_state[0]
            # add the experience to our replay buffer
            experience = Experience(state, done, action, reward, new_state)
            replay_buffer.add_experience(experience)

            # render the environment
            if render:
                env.render()

            # train on the experience
            if not done:
                if replay_buffer.mb_size < replay_buffer.num_experiences():
                    training_batch = replay_buffer.sample_minibatch()
                    loss = dqn_agent.train_on_batch(training_batch)

                    if verbose:
                        print(loss)

            ep_return += reward

            state = new_state

        # episode terminated by this point
        returns[ep] = ep_return
        print(f"Episode {ep} over. Total return: {ep_return}")

    plt.plot(returns)
    _ = plt.title("Agent total returns per episode (Training)"), plt.xlabel("Episode"), plt.ylabel("Return")

    plt.show()

    # print(returns)
    return dqn_agent, returns


def evaluate_agent_on_env(agent: DqnAgent, env: gym.Env, num_eval_episodes=100, render=True):
    # set the agent's random move probability to 0, so we can evaluate its policy exclusively.
    # agent MUST expose a set_epsilon method to control randomness, hence we type-hint that it is a DqnAgent
    agent.set_epsilon(0.0)

    eval_returns = np.zeros(num_eval_episodes)
    for eval_ep in range(num_eval_episodes):
        # reset our environment for this run
        state = env.reset()
        done = False
        ep_return = 0.

        while not done:
            # call the action wrapper to get an e-greedy action
            action = agent.action(state)

            # run the action on the environment and get the new info
            new_state, reward, done, info = env.step(action)

            # render the environment
            if render:
                env.render('human')

            ep_return += reward
            state = new_state

        # episode terminated by this point
        eval_returns[eval_ep] = ep_return
        print(f"Evaluation episode {eval_ep} over. Total return: {ep_return}")

    return eval_returns


def main():
    # a really simple example observation and action space. Importantly, num_observations is the size of a state
    #    and num_actions is the number of actions available in the environment.

    example_state = (0., 0., 1.)
    example_action_set = list(range(5))

    # we need to instruct the DQN Agent to not use convolutional layers, otherwise it will expect pixel-shaped inputs.
    agent = DqnAgent(obs_shape=(3, ), num_actions=len(example_action_set), qnet_conv_layer_params=None,
                     epsilon=1., rng_seed=150)
    print(f"Sampling a random action: {agent.action(example_state)}")

    agent.set_epsilon(0.)
    print(f"Sampling a greedy action: {agent.action(example_state)}")

    # train an agent on a given environment
    test_env = gym.envs.make('CarRacing-v2',
                             continuous=False, render_mode='human'
                             )

    trained_agent, returns = run_dqn_on_env(test_env, num_episodes=1000, render=True)

    # evaluate the agent on the same environment
    eval_returns = evaluate_agent_on_env(trained_agent, test_env, num_eval_episodes=250, render=False)
    plt.plot(eval_returns)
    plt.title("Agent total returns per episode (Evaluation)"), plt.xlabel("Eval. episode"), plt.ylabel("Returns")
    plt.show()

    # trained_agent.save_policy()


if __name__ == '__main__':
    main()
    # cartpole_test(num_episodes=1000, render=False)
