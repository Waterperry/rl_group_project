__author__ = 'tb791'

import gym
import numpy as np
import pygame
import tf_agents.networks.q_network

from keras.optimizers import Adam
from utils import distance_preprocess

from tf_agents.agents.dqn import dqn_agent as tf_dqn_agent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.metrics.tf_metrics import AverageReturnMetric
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from SoftTargetDqn import SoftTargetDqnAgent
from TargetDqnAgent import TargetDqnAgent
from gym.spaces import Discrete
from matplotlib import pyplot as plt
from UniformReplayBuffer import UniformReplayBuffer, Experience
from DqnAgent import DqnAgent


def run_dqn_on_env(env: gym.Env, num_episodes=150, render=True, verbose=False):
    assert type(env.action_space) == Discrete, "Can only use this DQN Agent on discrete action spaces for now."

    # reset the environment and get the initial state s0
    state = env.reset()
    if len(state) == 2 and type(state) == tuple:
        state = state[0]

    # create our DQN agent, passing it information about the environment's observation/action spec.
    dqn_agent = DqnAgent(state.shape, env.action_space.n, qnet_fc_layer_params=(256, 256, 128),
                         epsilon=0.2, gamma=0.95)

    replay_buffer = UniformReplayBuffer(max_length=10000, minibatch_size=32)

    if render:
        print("[WARN]: Rendering will slow down training. Are you sure you want to be rendering?")

    # while the episode isn't over, generate a new action on the state, perform that action, then train.
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        ep_return = 0.
        state = env.reset()

        done = truncated = False
        while not done or truncated:
            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                state = state[0]
                # call the action wrapper to get an e-greedy action
            action = dqn_agent.action(state)

            # run the action on the environment and get the new info
            # if you get a ValueError about unpacking values here, swap these lines around.
            # new_state, reward, done, info = env.step(action)
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
            if not done or truncated:
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


def run_tdqn_on_env(env: gym.Env, num_episodes=150, render=True, verbose=False):
    assert type(env.action_space) == Discrete, "Can only use this DQN Agent on discrete action spaces for now."

    # reset the environment and get the initial state s0
    state = env.reset()
    if len(state) == 2 and type(state) == tuple:
        state = state[0]
        state = distance_preprocess(state)

    # create our DQN agent, passing it information about the environment's observation/action spec.
    dqn_agent = TargetDqnAgent(state.shape, env.action_space.n, target_update_period=25, qnet_conv_layer_params=None,
                               qnet_fc_layer_params=(64, 64), epsilon=0.2, gamma=0.95)

    replay_buffer = UniformReplayBuffer(max_length=10000, minibatch_size=32)

    if render:
        print("[WARN]: Rendering will slow down training. Are you sure you want to be rendering?")

    # while the episode isn't over, generate a new action on the state, perform that action, then train.
    returns = np.zeros(num_episodes)
    for ep in range(num_episodes):
        ep_return = 0.
        state = env.reset()

        done, truncated = False, False
        while not (done or truncated):
            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                state = state[0]
                state = distance_preprocess(state)

            action = dqn_agent.action(state)

            # run the action on the environment and get the new info
            new_state, reward, done, truncated, info = env.step(action)

            # state is a tuple in CarRacing for some reason. just get the pixel-based observation.
            if len(state) == 2 and type(state) == tuple:
                new_state = new_state[0]

            new_state = distance_preprocess(new_state)
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
            new_state, reward, done, truncated, info = env.step(action)

            # render the environment
            if render:
                env.render()

            ep_return += reward
            state = new_state

        # episode terminated by this point
        eval_returns[eval_ep] = ep_return
        print(f"Evaluation episode {eval_ep} over. Total return: {ep_return}")

    return eval_returns


dqn_cartpole_layer_params = (32, 32)
dqn_cartpole_alpha = 1e-4
dqn_cartpole_target_update_tau = 0.1
dqn_cartpole_target_update_period = 15
dqn_cartpole_batch_size = 128


def cartpole_test(num_episodes=150, render=True, verbose=False):
    env = gym.make('CartPole-v1')

    # reset the environment and get the initial state s0
    state = env.reset()
    if len(state) == 2 and type(state) == tuple:
        state = state[0]

    # create our DQN agent, passing it information about the environment's observation/action spec.
    dqn_agent = SoftTargetDqnAgent(state.shape, env.action_space.n, qnet_conv_layer_params=None,
                                   target_update_period=dqn_cartpole_target_update_period,
                                   alpha=dqn_cartpole_alpha, qnet_fc_layer_params=dqn_cartpole_layer_params, epsilon=0.)

    replay_buffer = UniformReplayBuffer(max_length=20000, minibatch_size=dqn_cartpole_batch_size)

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
        if verbose:
            print(f"Episode {ep} over. Total return: {ep_return}")

    # plt.plot(returns)
    # _ = plt.title("Agent total returns per episode (Training)"), plt.xlabel("Episode"), plt.ylabel("Return")
    # plt.show()
    return dqn_agent, returns


def cartpole_tddqn_nstep_baseline(num_episodes=150, render=False, verbose=False):
    n_steps = 15
    if render:
        py_env = gym.envs.make('CartPole-v1', render_mode='human')
    else:
        py_env = gym.envs.make('CartPole-v1')
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(py_env))

    q_network = tf_agents.networks.q_network.QNetwork(
        tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=dqn_cartpole_layer_params)
    agent = tf_dqn_agent.DdqnAgent(tf_env.time_step_spec(), tf_env.action_spec(), q_network,
                                   Adam(learning_rate=dqn_cartpole_alpha), target_update_tau=0.5, n_step_update=n_steps,
                                   target_update_period=dqn_cartpole_target_update_period, epsilon_greedy=0.)

    reward = AverageReturnMetric()
    rb = TFUniformReplayBuffer(agent.collect_data_spec, 1, max_length=20000)
    rb_iter = iter(rb.as_dataset(dqn_cartpole_batch_size, num_steps=n_steps+1))
    driver = DynamicStepDriver(tf_env, agent.policy, [reward, rb.add_batch])

    rewards = np.zeros(num_episodes)
    for ep in range(num_episodes):
        step = tf_env.reset()

        while not step.is_last():
            step, _ = driver.run(step)
            if render:
                tf_env.render(mode='human')
                pygame.event.get()
            if rb.num_frames() > dqn_cartpole_batch_size:
                agent.train(next(rb_iter)[0])

        rewards[ep] = reward.result()
        if verbose:
            print(f"{ep}, {reward.result().numpy()}")
        reward.reset()
    return rewards


def cartpole_tddqn_baseline(num_episodes=150, render=False, verbose=False):
    if render:
        py_env = gym.envs.make('CartPole-v1', render_mode='human')
    else:
        py_env = gym.envs.make('CartPole-v1')
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(py_env))

    q_network = tf_agents.networks.q_network.QNetwork(
        tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=dqn_cartpole_layer_params)
    agent = tf_dqn_agent.DdqnAgent(tf_env.time_step_spec(), tf_env.action_spec(), q_network,
                                   Adam(learning_rate=dqn_cartpole_alpha), target_update_tau=0.5,
                                   target_update_period=dqn_cartpole_target_update_period, epsilon_greedy=0.)

    reward = AverageReturnMetric()
    rb = TFUniformReplayBuffer(agent.collect_data_spec, 1, max_length=20000)
    rb_iter = iter(rb.as_dataset(dqn_cartpole_batch_size, num_steps=2))
    driver = DynamicStepDriver(tf_env, agent.policy, [reward, rb.add_batch])

    rewards = np.zeros(num_episodes)
    for ep in range(num_episodes):
        step = tf_env.reset()

        while not step.is_last():
            step, _ = driver.run(step)
            if render:
                tf_env.render(mode='human')
                pygame.event.get()
            if rb.num_frames() > dqn_cartpole_batch_size:
                agent.train(next(rb_iter)[0])

        rewards[ep] = reward.result()
        if verbose:
            print(f"{ep}, {reward.result().numpy()}")
        reward.reset()
    return rewards


def cartpole_ddqn_baseline(num_episodes=150, render=False, verbose=False):
    if render:
        py_env = gym.envs.make('CartPole-v1', render_mode='human')
    else:
        py_env = gym.envs.make('CartPole-v1')
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(py_env))

    q_network = tf_agents.networks.q_network.QNetwork(
        tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=dqn_cartpole_layer_params)
    agent = tf_dqn_agent.DdqnAgent(tf_env.time_step_spec(), tf_env.action_spec(), q_network,
                                   Adam(learning_rate=dqn_cartpole_alpha), epsilon_greedy=0.)

    reward = AverageReturnMetric()
    rb = TFUniformReplayBuffer(agent.collect_data_spec, 1, max_length=20000)
    rb_iter = iter(rb.as_dataset(dqn_cartpole_batch_size, num_steps=2))
    driver = DynamicStepDriver(tf_env, agent.policy, [reward, rb.add_batch])

    rewards = np.zeros(num_episodes)
    for ep in range(num_episodes):
        step = tf_env.reset()

        while not step.is_last():
            step, _ = driver.run(step)
            if render:
                tf_env.render(mode='human')
                pygame.event.get()
            if rb.num_frames() > dqn_cartpole_batch_size:
                agent.train(next(rb_iter)[0])

        rewards[ep] = reward.result()
        if verbose:
            print(f"{ep}, {reward.result().numpy()}")
        reward.reset()
    return rewards


def cartpole_baseline(num_episodes=150, render=False, verbose=False):
    if render:
        py_env = gym.envs.make('CartPole-v1', render_mode='human')
    else:
        py_env = gym.envs.make('CartPole-v1')
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(py_env))

    q_network = tf_agents.networks.q_network.QNetwork(
        tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=dqn_cartpole_layer_params)
    agent = tf_dqn_agent.DqnAgent(tf_env.time_step_spec(), tf_env.action_spec(), q_network,
                                  target_update_tau=dqn_cartpole_target_update_tau,
                                  target_update_period=dqn_cartpole_target_update_period,
                                  optimizer=Adam(learning_rate=dqn_cartpole_alpha), epsilon_greedy=0.)

    reward = AverageReturnMetric()
    rb = TFUniformReplayBuffer(agent.collect_data_spec, 1, max_length=20000)
    rb_iter = iter(rb.as_dataset(dqn_cartpole_batch_size, num_steps=2))
    driver = DynamicStepDriver(tf_env, agent.policy, [reward, rb.add_batch])

    rewards = np.zeros(num_episodes)
    for ep in range(num_episodes):
        step = tf_env.reset()

        while not step.is_last():
            step, _ = driver.run(step)
            if render:
                tf_env.render(mode='human')
                pygame.event.get()
            if rb.num_frames() > dqn_cartpole_batch_size:
                agent.train(next(rb_iter)[0])

        rewards[ep] = reward.result()
        if verbose:
            print(f"{ep}, {reward.result().numpy()}")
        reward.reset()
    return rewards


def cartpole_tdqn_baseline(num_episodes=150, render=False, verbose=False):
    if render:
        py_env = gym.envs.make('CartPole-v1', render_mode='human')
    else:
        py_env = gym.envs.make('CartPole-v1')
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(py_env))

    q_network = tf_agents.networks.q_network.QNetwork(
        tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=dqn_cartpole_layer_params)
    agent = tf_dqn_agent.DqnAgent(tf_env.time_step_spec(), tf_env.action_spec(), q_network,
                                  Adam(learning_rate=dqn_cartpole_alpha),
                                  target_update_tau=dqn_cartpole_target_update_tau,
                                  target_update_period=dqn_cartpole_target_update_period, epsilon_greedy=0.)

    reward = AverageReturnMetric()
    rb = TFUniformReplayBuffer(agent.collect_data_spec, 1, max_length=20000)
    rb_iter = iter(rb.as_dataset(dqn_cartpole_batch_size, num_steps=2))
    driver = DynamicStepDriver(tf_env, agent.policy, [reward, rb.add_batch])

    rewards = np.zeros(num_episodes)
    for ep in range(num_episodes):
        step = tf_env.reset()

        while not step.is_last():
            step, _ = driver.run(step)
            if render:
                tf_env.render(mode='human')
                pygame.event.get()
            if rb.num_frames() > dqn_cartpole_batch_size:
                agent.train(next(rb_iter)[0])

        rewards[ep] = reward.result()
        if verbose:
            print(f"{ep}, {reward.result().numpy()}")
        reward.reset()
    return rewards


def cartpole_evaluate_versus_baseline(num_eps=100, num_runs=25) -> (np.ndarray, np.ndarray):
    baselines = np.zeros((num_runs, num_eps))
    ours = np.zeros((num_runs, num_eps))
    for i in range(num_runs):
        print(f"Start run {i}")
        baseline_rewards = cartpole_baseline(num_episodes=num_eps, verbose=False)
        baselines[i] = baseline_rewards
        _, our_rewards = cartpole_test(num_episodes=num_eps, render=False, verbose=False)
        ours[i] = our_rewards

    baseline_rewards = np.mean(baselines, axis=0)
    our_rewards = np.mean(ours, axis=0)
    plt.plot(baseline_rewards, label='baseline')
    plt.plot(our_rewards, label='ours')
    plt.legend()
    plt.show()

    return baselines, ours


if __name__ == '__main__':
    # n_eps = 300
    # raw_rewards = cartpole_baseline(n_eps)
    # tgt_rewards = cartpole_tdqn_baseline(n_eps)
    # ddqn_rewards = cartpole_ddqn_baseline(n_eps)
    # tddqn_rewards = cartpole_tddqn_baseline(n_eps, verbose=True)
    # ddqn_nstep_rewards = cartpole_tddqn_nstep_baseline(n_eps)

    # for idx in range(raw_rewards.shape[0]):
    #     print(f'{raw_rewards[idx]}, {tgt_rewards[idx]}, {ddqn_rewards[idx]}, {tddqn_rewards[idx]}')

    # plt.plot(raw_rewards, label='Vanilla DQN')
    # plt.plot(tgt_rewards, label='Target DQN')
    # plt.plot(ddqn_rewards, label='Vanilla DDQN')
    # plt.plot(tddqn_rewards, label='Target DDQN')
    # plt.ylim(0, 100)
    # plt.plot(ddqn_nstep_rewards, label='N-step DDQN')
    # plt.legend()
    # plt.show()

    cartpole_evaluate_versus_baseline(num_runs=5)
    # exit(0)

    # train an agent on a given environment
    # test_env = gym.envs.make('CarRacing-v2', continuous=False, render_mode='human')
    #
    # trained_agent, trained_returns = run_tdqn_on_env(test_env, num_episodes=1000, render=True)
    # plt.plot(trained_returns)
    # plt.show()

    # evaluate the agent on the same environment
    # eval_returns = evaluate_agent_on_env(trained_agent, test_env, num_eval_episodes=250, render=False)
    # plt.plot(eval_returns)
    # plt.title("Agent total returns per episode (Evaluation)"), plt.xlabel("Eval. episode"), plt.ylabel("Returns")
    # plt.show()

    # trained_agent.save_policy()
