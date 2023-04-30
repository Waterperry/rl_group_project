from keras.optimizers import Adam
from pygame.event import get as pygame_event_get
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from gym.envs import make as gym_envs_make
from tf_agents.environments import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.utils.common import element_wise_squared_loss
from tf_agents.metrics.tf_metrics import AverageReturnMetric
from keras.layers.preprocessing.image_preprocessing import Rescaling


# don't worry if these functions don't work on your device, I absolutely butchered my tf-agents install
#    to get these to work...
def target_dqn_baseline(num_episodes=100, target_update_period=25, target_update_tau=0.5, render=False):
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
    step = tf_env.reset()

    for episode in range(num_episodes):
        while not step.is_last():
            step, _ = driver.run(step)
            if episode > 0 or replay_buffer.num_frames() > batch_size:
                _ = t_dqn.train(next(train_iter)[0])

            if render:
                tf_env.render(mode='human')
                pygame_event_get()

        print(f"{episode}, {reward_metric.result()}")
        reward_metric.reset()


if __name__ == '__main__':
    target_dqn_baseline(render=True)
