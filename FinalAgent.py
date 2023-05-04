__author__ = 'tb791'

import pygame
import tensorflow as tf

import SoftTargetDqn
import UniformReplayBuffer
import utils
from DoubleDqnAgent import DoubleDqnAgent
from agents import NStepDdqnAgent, Experience
import numpy as np
from matplotlib import pyplot as plt
import gym


class FinalAgent(NStepDdqnAgent):
    def __init__(self, obs_shape, num_actions,
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
                 target_update_tau=0.5
                 ):
        super().__init__(obs_shape=obs_shape, num_actions=num_actions, alpha=alpha, epsilon=epsilon, gamma=gamma,
                         qnet_fc_layer_params=qnet_fc_layer_params, qnet_conv_layer_params=qnet_conv_layer_params,
                         rng_seed=rng_seed, debug=debug, update_interval=update_interval, epsilon_decay=epsilon_decay,
                         decay_factor=decay_factor, preprocessing_layer=preprocessing_layer, n_steps=n_steps,
                         target_update_tau=target_update_tau)


def run_training_loop(num_episodes, dqn_agent, env, replay_buffer, render):
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        state = tf.squeeze(utils.tf_distance_preprocess(env.reset()[0]))
        done, truncated = False, False
        ep_return = 0.

        while not done and not truncated:
            action = dqn_agent.action(state)
            new_state, reward, done, truncated, info = env.step(action)

            new_state = tf.squeeze(utils.tf_distance_preprocess(new_state))
            done = done or truncated

            replay_buffer.add_experience(Experience(state, done, action, reward, new_state))

            if render:
                env.render()
                pygame.event.get()

            if replay_buffer.mb_size < replay_buffer.num_experiences():
                train_data = replay_buffer.sample_minibatch()
                dqn_agent.train_on_batch(train_data)

            ep_return += reward
            state = new_state

        returns[ep] = ep_return
        print(f"{ep}, {ep_return}")

    return dqn_agent, returns


def run_nstep_training_loop(num_episodes, dqn_agent, env, replay_buffer, render, n_steps):
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        state = tf.squeeze(utils.tf_distance_preprocess(env.reset()[0]))
        done, truncated = False, False
        ep_return = 0.
        buf = []
        act_step = 0
        grass_count = 0
        while not done and not truncated:
            act_step += 1
            action = dqn_agent.action(state)
            new_state, reward, done, truncated, info = env.step(action)

            new_state = tf.squeeze(utils.tf_distance_preprocess(new_state))
            if tf.reduce_mean(new_state) < 5 and act_step > 40:
                grass_count += 1
                if grass_count > 5:
                    done = True
                    reward = -10
            else:
                grass_count = 0
            done = done or truncated

            # Environment zooms in from out. This can mess up our training data.
            if act_step > 40:
                new_experience = Experience(state, done, action, reward, new_state)
                buf.append(new_experience)

            if len(buf) >= n_steps:
                # compute the n_step reward times our gamma
                n_step_reward = 0

                # for every subsequent experience in the buffer, if it's not terminal, incorporate it
                for idx, exper in enumerate(buf[1:]):
                    if exper.is_terminal():
                        break
                    n_step_reward += exper.reward * (dqn_agent.get_gamma() ** (idx + 1))

                # create the n-step experience and add it to the replay buffer
                n_step_experience = buf[0]
                n_step_experience.reward = n_step_reward
                replay_buffer.add_experience(n_step_experience)

                # remove it from the buffer, so we can add our new one.
                buf.pop(0)

            if render:
                env.render()
                pygame.event.get()

            if replay_buffer.mb_size < replay_buffer.num_experiences():
                train_data = replay_buffer.sample_minibatch()
                dqn_agent.train_on_batch(train_data)

            ep_return += reward
            state = new_state

        returns[ep] = ep_return
        print(f"{ep}, {ep_return}")

    return dqn_agent, returns


def run_nstep_training_loop_no_preprop(num_episodes, dqn_agent, env, replay_buffer, render, n_steps):
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        state, _ = env.reset()
        done, truncated = False, False
        ep_return = 0.
        buf = []
        while not done and not truncated:
            action = dqn_agent.action(state)
            new_state, reward, done, truncated, info = env.step(action)

            done = done or truncated

            new_experience = Experience(state, done, action, reward, new_state)
            buf.append(new_experience)

            if len(buf) >= n_steps:
                # compute the n_step reward times our gamma
                n_step_reward = 0

                # for every subsequent experience in the buffer, if it's not terminal, incorporate it
                for idx, exper in enumerate(buf[1:]):
                    if exper.is_terminal():
                        break
                    n_step_reward += exper.reward * (dqn_agent.get_gamma() ** (idx + 1))

                # create the n-step experience and add it to the replay buffer
                n_step_experience = buf[0]
                n_step_experience.reward = n_step_reward
                replay_buffer.add_experience(n_step_experience)

                # remove it from the buffer, so we can add our new one.
                buf.pop(0)

            if render:
                env.render()
                pygame.event.get()

            if replay_buffer.mb_size < replay_buffer.num_experiences():
                train_data = replay_buffer.sample_minibatch()
                dqn_agent.train_on_batch(train_data)

            ep_return += reward
            state = new_state

        returns[ep] = ep_return
        print(f"{ep}, {ep_return}")

    return dqn_agent, returns



def run_final_agent(num_episodes=150, render=True):
    env = gym.envs.make('CarRacing-v2', render_mode='human', continuous=False)
    n_step = True
    n_steps = 15
    # reset the environment and get the initial state s0
    state = env.reset()
    if len(state) == 2 and type(state) == tuple:
        state = state[0]
        state = tf.squeeze(utils.tf_distance_preprocess(state))

    # create our DQN agent, passing it information about the environment's observation/action spec.
    dqn_agent = FinalAgent(state.shape, env.action_space.n, qnet_fc_layer_params=(64, 64),
                           target_update_tau=0.2, epsilon_decay=False, update_interval=50,
                           qnet_conv_layer_params=None, preprocessing_layer=False, epsilon=0.1, gamma=0.99,
                           n_steps=n_steps, alpha=1e-4)

    # replay_buffer = PrioritizedReplayBuffer(buffer_size=10000, batch_size=32, n_steps=5)
    replay_buffer = UniformReplayBuffer.UniformReplayBuffer(max_length=20000)
    if render:
        print("[WARN]: Rendering will slow down training. Are you sure you want to be rendering?")

    # while the episode isn't over, generate a new action on the state, perform that action, then train.
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        ep_return = 0.
        state = env.reset()
        done, truncated = False, False
        temp_buffer = []
        while not (done or truncated):
            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                state = state[0]
                state = tf.squeeze(utils.tf_distance_preprocess(state))

            # call the action wrapper to get an e-greedy action
            action = dqn_agent.action(state)
            new_state, reward, done, truncated, info = env.step(action)

            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                new_state = new_state[0]
            new_state = tf.squeeze(utils.tf_distance_preprocess(new_state))

            # run the action on the environment and get the new info
            # if you get a ValueError about unpacking values here, swap these lines around.
            # new_state, reward, done, info = env.step(action)
            experience = Experience(state, done, action, reward, new_state)

            if n_step:
                temp_buffer.append(experience)
                if len(temp_buffer) >= n_steps and n_step:
                    n_step_rewards = [exp.reward for exp in temp_buffer]
                    n_step_experience = Experience(state, done, action,
                                                   n_step_rewards, new_state, n_step_rewards)
                    replay_buffer.add_experience(n_step_experience)
                    temp_buffer.clear()
            else:
                # add the experience to our replay buffer
                replay_buffer.add_experience(experience)

            # render the environment
            if render:
                env.render()
                pygame.event.get()

            # train on the experience
            if not done or truncated:
                if replay_buffer.mb_size < replay_buffer.num_experiences():
                    train_exp = replay_buffer.sample_minibatch()
                    dqn_agent.train_on_batch(train_exp)
                # if replay_buffer.batch_size < replay_buffer.num_experiences():
                #     training_batch, (importance, indices) = replay_buffer.get_batch()
                #     loss = dqn_agent.train_on_batch(training_batch)
                #     loss = loss * len(indices)
                #     replay_buffer.update_priorities(indices, loss)
                #     if verbose:
                #         print(loss)

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


def run_soft_dqn_agent(num_episodes=150, render=True):
    env = gym.envs.make('CarRacing-v2', render_mode='human', continuous=False)
    state = tf.squeeze(utils.tf_distance_preprocess(env.reset()[0]))

    dqn_agent = SoftTargetDqn.SoftTargetDqnAgent(state.shape, env.action_space.n, qnet_fc_layer_params=(64, 64),
                                                 target_update_tau=0.1, target_update_period=50,
                                                 qnet_conv_layer_params=None, epsilon=0.1, alpha=1e-4)

    replay_buffer = UniformReplayBuffer.UniformReplayBuffer(max_length=20000, minibatch_size=64)

    dqn_agent, returns = run_training_loop(num_episodes, dqn_agent, env, replay_buffer, render)
    plt.plot(returns)
    plt.title("Agent return per episode"), plt.xlabel("Episode"), plt.ylabel("Return")
    plt.show()

    return dqn_agent, returns


def run_soft_ddqn_agent(num_episodes=150, render=True):
    env = gym.envs.make('CarRacing-v2', render_mode='human', continuous=False)
    state = tf.squeeze(utils.tf_distance_preprocess(env.reset()[0]))

    dqn_agent = DoubleDqnAgent(state.shape, env.action_space.n, qnet_fc_layer_params=(64, 64),
                               target_update_tau=0.1, target_update_period=50,
                               qnet_conv_layer_params=None, epsilon=0.1, alpha=1e-4)

    replay_buffer = UniformReplayBuffer.UniformReplayBuffer(max_length=20000, minibatch_size=64)

    dqn_agent, returns = run_training_loop(num_episodes, dqn_agent, env, replay_buffer, render)
    plt.plot(returns)
    plt.title("Agent return per episode"), plt.xlabel("Episode"), plt.ylabel("Return")
    plt.show()
    return dqn_agent, returns


def run_soft_n_step_ddqn_agent(num_episodes=150, render=True):
    env = gym.envs.make('CarRacing-v2', render_mode='human', continuous=False)
    state = tf.squeeze(utils.tf_distance_preprocess(env.reset()[0]))

    dqn_agent = DoubleDqnAgent(state.shape, env.action_space.n, qnet_fc_layer_params=(64, 64),
                               target_update_tau=0.5, target_update_period=25,
                               qnet_conv_layer_params=None, epsilon=0.1, alpha=1e-4)

    replay_buffer = UniformReplayBuffer.UniformReplayBuffer(max_length=20000, minibatch_size=64)

    dqn_agent, returns = run_nstep_training_loop(num_episodes, dqn_agent, env, replay_buffer, render, n_steps=10)
    plt.plot(returns)
    plt.title("Agent return per episode"), plt.xlabel("Episode"), plt.ylabel("Return")
    plt.show()
    return dqn_agent, returns


def cartpole_n_step_soft(num_episodes=500, render=True):
    env = gym.envs.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()

    dqn_agent = DoubleDqnAgent(state.shape, env.action_space.n, qnet_fc_layer_params=(64, 64),
                               target_update_tau=0.5, target_update_period=25,
                               qnet_conv_layer_params=None, epsilon=0.25)

    replay_buffer = UniformReplayBuffer.UniformReplayBuffer(max_length=20000, minibatch_size=64)

    dqn_agent, returns = run_nstep_training_loop_no_preprop(num_episodes, dqn_agent, env,
                                                            replay_buffer, render, n_steps=10)
    plt.plot(returns)
    plt.title("Agent return per episode"), plt.xlabel("Episode"), plt.ylabel("Return")
    plt.show()
    return dqn_agent, returns


if __name__ == '__main__':
    # final_agent, returns = run_final_agent(300)
    # final_agent.save_policy()
    # final_agent, returns = run_soft_ddqn_agent(100)
    # final_agent.save_policy()
    final_agent, returns = run_soft_n_step_ddqn_agent(100, render=True)
    final_agent.save_policy()
    # final_agent, returns = cartpole_n_step_soft(500)
