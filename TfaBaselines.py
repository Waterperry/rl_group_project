__author__ = 'tb791'

import pygame
import tf_agents.specs
from keras.layers import Lambda
from tf_agents.agents.ddpg import actor_network, critic_network, ddpg_agent
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tensorflow import int64 as tf_int64
import tensorflow as tf

import utils
from utils import tf_distance_preprocess
from keras.optimizers import Adam
from pygame.event import get as pygame_event_get
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from gym.envs import make as gym_envs_make
from tf_agents.environments import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.utils.common import element_wise_squared_loss
from tf_agents.metrics.tf_metrics import AverageReturnMetric
from keras.layers.preprocessing.image_preprocessing import Rescaling
from tf_agents.trajectories import time_step as ts, Trajectory


# don't worry if these functions don't work on your device, I absolutely butchered my tf-agents install
#    to get these to work...
def target_dqn_baseline(num_episodes=100, target_update_period=25, target_update_tau=0.5, render=False):
    """
    Baseline DQN + Exp. Replay + Target Network on CarRacing.
    :param num_episodes:
    :param target_update_period:
    :param target_update_tau:
    :param render:
    :return:
    """
    batch_size = 64

    # set up the py environment, and wrap it then convert to tensorflow-agents environment
    py_env = gym_envs_make('CarRacing-v2', continuous=False, render_mode='human')
    tf_env = TFPyEnvironment(wrap_env(py_env))

    # create the q-network and target-dqn agent
    q_net = QNetwork(tf_env.observation_spec(), tf_env.action_spec(), preprocessing_layers=Rescaling(1./255.),
                     conv_layer_params=((32, 2, 2), (64, 2, 2), (128, 2, 2)), fc_layer_params=(256, 256, 128))
    t_dqn = DqnAgent(tf_env.time_step_spec(), tf_env.action_spec(), q_net, optimizer=Adam(),
                     epsilon_greedy=0.1, target_update_period=target_update_period, target_update_tau=target_update_tau,
                     td_errors_loss_fn=element_wise_squared_loss)

    # create the replay buffer, reward metric, and step driver
    reward_metric = AverageReturnMetric()
    replay_buffer = TFUniformReplayBuffer(t_dqn.collect_data_spec, batch_size=1, max_length=10000)
    driver = DynamicStepDriver(tf_env, t_dqn.collect_policy, observers=[reward_metric, replay_buffer.add_batch])

    # create the training data iterator for the agent, as well as resetting the environment to get s0
    train_iter = iter(replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2))

    for episode in range(num_episodes):
        step = tf_env.reset()
        reward_metric.reset()

        while not step.is_last():
            step, _ = driver.run(step)
            if episode > 0 or replay_buffer.num_frames() > batch_size:
                _ = t_dqn.train(next(train_iter)[0])

            if render:
                tf_env.render(mode='human')
                pygame_event_get()

        print(f"{episode}, {reward_metric.result()}")


def random_baseline(num_episodes=100, target_update_period=25, target_update_tau=0.5, render=False):
    """
    Baseline DQN + Exp. Replay + Target Network on CarRacing.
    :param num_episodes:
    :param target_update_period:
    :param target_update_tau:
    :param render:
    :return:
    """
    batch_size = 64

    # set up the py environment, and wrap it then convert to tensorflow-agents environment
    py_env = gym_envs_make('CarRacing-v2', continuous=False, render_mode='human')
    tf_env = TFPyEnvironment(wrap_env(py_env))

    # create the q-network and target-dqn agent
    t_dqn = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

    # create the replay buffer, reward metric, and step driver
    reward_metric = AverageReturnMetric()
    # replay_buffer = TFUniformReplayBuffer(t_dqn.collect_data_spec, batch_size=1, max_length=10000)
    driver = DynamicStepDriver(tf_env, t_dqn.collect_policy, observers=[reward_metric])

    for episode in range(num_episodes):
        step = tf_env.reset()
        reward_metric.reset()

        while not step.is_last():
            step, _ = driver.run(step)

            if render:
                tf_env.render(mode='human')
                pygame_event_get()

        print(f"{episode}, {reward_metric.result()}")


def final_dqn_baseline(num_episodes=100, target_update_period=25, target_update_tau=0.5, render=False):
    """
    Baseline DQN + Exp. Replay + Target + n-step + DDQN
    :param num_episodes:
    :param target_update_period:
    :param target_update_tau:
    :param render:
    :return:
    """
    batch_size = 64
    n_steps = 15

    # set up the py environment, and wrap it then convert to tensorflow-agents environment
    py_env = gym_envs_make('CarRacing-v2', continuous=False, render_mode='human')
    tf_env = TFPyEnvironment(wrap_env(py_env))

    # create the q-network and target-dqn agent
    q_net = QNetwork(tf_env.observation_spec(), tf_env.action_spec(),
                     preprocessing_layers=Lambda(tf_distance_preprocess), fc_layer_params=(64, 64))
    t_dqn = DdqnAgent(tf_env.time_step_spec(), tf_env.action_spec(), q_net, optimizer=Adam(),
                      epsilon_greedy=0.1, target_update_period=target_update_period, target_update_tau=target_update_tau,
                      td_errors_loss_fn=element_wise_squared_loss, n_step_update=n_steps)

    # create the replay buffer, reward metric, and step driver
    reward_metric = AverageReturnMetric()
    replay_buffer = TFUniformReplayBuffer(t_dqn.collect_data_spec, batch_size=1, max_length=10000)
    driver = DynamicStepDriver(tf_env, t_dqn.collect_policy, observers=[reward_metric, replay_buffer.add_batch])

    # create the training data iterator for the agent, as well as resetting the environment to get s0
    train_iter = iter(replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=n_steps+1))

    for episode in range(num_episodes):
        step = tf_env.reset()
        reward_metric.reset()

        while not step.is_last():
            step, _ = driver.run(step)
            if episode > 0 or replay_buffer.num_frames() > batch_size:
                _ = t_dqn.train(next(train_iter)[0])

            if render:
                tf_env.render(mode='human')
                pygame_event_get()

        print(f"{episode}, {reward_metric.result()}")


def final_ac_baseline(num_episodes=100, render=False):
    batch_size = 64
    target_update_period = 25
    target_update_tau = 0.5
    alpha = 5e-4
    py_env = gym_envs_make('CarRacing-v2', render_mode='human')
    tf_env = TFPyEnvironment(wrap_env(py_env))

    step: ts.TimeStep = tf_env.reset()

    obs_pp = utils.tf_distance_preprocess(step.observation)
    pp_spec = tf_agents.specs.TensorSpec(obs_pp.shape, dtype=tf_int64)

    env_ts_spec: ts.TimeStep = tf_env.time_step_spec()
    env_ts_spec = ts.TimeStep(env_ts_spec.step_type, env_ts_spec.reward, env_ts_spec.discount, pp_spec)

    actor_net = actor_network.ActorNetwork(pp_spec, tf_env.action_spec(), fc_layer_params=(64, 64),)
    critic_net = critic_network.CriticNetwork((pp_spec, tf_env.action_spec()), joint_fc_layer_params=(64, 64))

    agent = ddpg_agent.DdpgAgent(env_ts_spec, tf_env.action_spec(), actor_net, critic_net,
                                 Adam(), Adam(), td_errors_loss_fn=element_wise_squared_loss)

    replay_buffer = TFUniformReplayBuffer(agent.collect_data_spec, batch_size=1, max_length=int(1e5))
    train_iter = iter(replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2))

    for episode in range(num_episodes):
        ep_return = 0.
        step = tf_env.reset()
        while not step.is_last():
            pp_step = ts.TimeStep(step.step_type, step.reward, step.discount, tf_distance_preprocess(step.observation))
            action, _, _ = agent.collect_policy.action(pp_step)
            next_step = tf_env.step(action)
            pp_next_step = ts.TimeStep(next_step.step_type, next_step.reward, next_step.discount,
                                       tf_distance_preprocess(next_step.observation))

            train_traj = Trajectory(pp_step.step_type, tf.expand_dims(pp_step.observation, axis=0),
                                    action, (), pp_next_step.step_type,
                                    pp_next_step.reward, pp_next_step.discount)
            replay_buffer.add_batch(train_traj)

            if episode > 0 or replay_buffer.num_frames() > batch_size:
                agent.train(next(train_iter)[0])

            if render:
                tf_env.render()
                pygame.event.get()

            ep_return += step.reward

            step = next_step
        print(f"{episode}, {ep_return}")





if __name__ == '__main__':
    # final_dqn_baseline(render=True)
    # random_baseline(300, render=True)
    final_ac_baseline(100, render=True)
