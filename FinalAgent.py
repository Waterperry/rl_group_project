import pygame
import tensorflow as tf
import utils
from agents import NStepDdqnAgent, Experience
from PERbuffer import PrioritizedReplayBuffer
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


def run_final_agent(num_episodes=150, render=True, verbose=False):
    env = gym.envs.make('CarRacing-v2', render_mode='human', continuous=False)
    n_step = True
    # reset the environment and get the initial state s0
    state = env.reset()
    if len(state) == 2 and type(state) == tuple:
        state = state[0]
        state = tf.squeeze(utils.tf_distance_preprocess(state))

    # create our DQN agent, passing it information about the environment's observation/action spec.
    dqn_agent = FinalAgent(state.shape, env.action_space.n, qnet_fc_layer_params=(256, 256, 128),
                           qnet_conv_layer_params=None, preprocessing_layer=False, epsilon=0.2, gamma=0.95)

    replay_buffer = PrioritizedReplayBuffer(buffer_size=10000, batch_size=32, n_steps=5)

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
                if len(temp_buffer) >= 5 and n_step:
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
                if replay_buffer.batch_size < replay_buffer.num_experiences():
                    training_batch, (importance, indices) = replay_buffer.get_batch()
                    loss = dqn_agent.train_on_batch(training_batch)
                    loss = loss * len(indices)
                    replay_buffer.update_priorities(indices, loss)
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


if __name__ == '__main__':
    run_final_agent(1)
