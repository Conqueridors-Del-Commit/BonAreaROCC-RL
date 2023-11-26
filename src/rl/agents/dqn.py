import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


from src.rl.environment.environment import EnvironmentBuilder

class CNNFeaturesPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs, ):
        # Pasa los parámetros adicionales a la clase base
        super(CNNFeaturesPolicy, self).__init__(*args, **kwargs, features_extractor_class=CNNFeatures)

class CNNFeatures(BaseFeaturesExtractor):
    def __init__(self, ob_space, ac_space, mask_function=None, **kwargs):
        super(CNNFeatures, self).__init__(ob_space, ac_space.n)
        self.mask_function = mask_function

        # convolution layers initialization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # full connected layers initialization
        self.fc_aux = nn.Linear(len(ob_space) - 1, 256)
        self.fc_combined = nn.Linear(64 * 7 * 7 + 256, ac_space.n)

        # activation function ReLU
        self.relu = nn.ReLU()

    def forward(self, obs, state=None, mask=None):
        # Split the observation (TODO use a feature extractor)
        image_obs = obs['cells']
        aux_obs = np.array([value for var, value in obs if var != 'cells'])

        # Process the cells using the convolution layers
        x = self.relu(self.conv1(image_obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # flatten the output of the convolution layers
        x = x.view(x.size(0), -1)

        # Process the aux obervation input data
        aux_x = self.relu(self.fc_aux(aux_obs))

        # concat two networks outputs
        combined_features = torch.cat((x, aux_x), dim=1)

        # pass the concatenation output to a full connected layer
        output = self.fc_combined(combined_features)

        return output, None

class CustomDQN(DQN):
    def __init__(self, *args, mask_function=None, **kwargs):
        super(CustomDQN, self).__init__(*args, **kwargs)
        self.mask_function = mask_function

    def _sample_action(self, learning_starts, action_noise, n_envs=1):
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            valid_actions = self.mask_function()
            valid_actions = [i for i in range(self.action_space.n) if valid_actions[i]]
            unscaled_action = np.array([random.choice(valid_actions) for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        buffer_action = unscaled_action
        action = buffer_action
        return action, buffer_action


if __name__ == '__main__':
    env = EnvironmentBuilder(
        ticket_path='data/data/test_ticket.csv',
        planogram_csv_path='data/data/planogram_table.csv',
        customer_properties_path='data/data/hackathon_customers_properties.csv',
        grouping_path='data/data/article_group.json',
        obs_mode=1,
        reward_mode=1).build()
    model = CustomDQN(
        CNNFeaturesPolicy,
        env,
        verbose=1,
        mask_function=env.get_valid_actions,
        policy_kwargs=dict(
           features_extractor_kwargs=dict(
                ac_space=env.action_space,
            )
        )
    )

    model.learn(total_timesteps=10000, log_interval=4)