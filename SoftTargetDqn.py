__author__ = "mmea21"
from TargetDqnAgent import TargetDqnAgent


class SoftTargetDqnAgent(TargetDqnAgent):
    def __init__(self,

                 obs_shape,  # size of the state |s|
                 num_actions,  # number of actions in the environment
                 target_update_tau=0.5,
                 target_update_period=25,  # number of train steps to update target network after
                 alpha=1e-3,  # AdaM learning rate
                 epsilon=0.1,  # random move probability
                 gamma=0.99,  # discount factor
                 qnet_fc_layer_params=(128, 64),  # neuron counts for the fully-connected layers of the Q Network
                 qnet_conv_layer_params=(32, 64, 128),  # filter counts for convolutional layers of the Q network
                 rng_seed: int = None,  # seed to RNG (optional, for debugging really)
                 debug: bool = False  # enable debugging mode, useful for stack traces in tensorflow functions
                 ):
        super().__init__(obs_shape, num_actions, alpha=alpha, epsilon=epsilon, gamma=gamma,
                         qnet_fc_layer_params=qnet_fc_layer_params, qnet_conv_layer_params=qnet_conv_layer_params,
                         rng_seed=rng_seed, debug=debug)

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._target_update_counter = 0

    def update_target(self):
        qnet_weights = self._qnet.get_weights()
        target_weights = self._tnet.get_weights()

        # Find difference between qnet_weights and target_weights, times by tau (0.5) and add to target_weights
        modified_qnet_weights = [self._target_update_tau * i for i in qnet_weights]
        modified_target_weights = [(1 - self._target_update_tau) * j for j in target_weights]
        weights = [modified_target_weights[x] + modified_qnet_weights[x] for x in range(len(modified_target_weights))]
        self._tnet.set_weights(weights)  # copy the Q-Network weights to target
        self._target_update_counter = 0
