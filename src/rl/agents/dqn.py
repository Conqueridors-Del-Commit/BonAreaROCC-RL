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

        # apply masking
        if self.mask_function:
            mask = self.mask_function(obs)
            output = output * mask
        return output, None


if __name__ == '__main__':
    env = EnvironmentBuilder(
        ticket_path='data/data/test_ticket.csv',
        planogram_csv_path='data/data/planogram_table.csv',
        customer_properties_path='data/data/hackathon_customers_properties.csv',
        grouping_path='data/data/article_group.json',
        obs_mode=1,
        reward_mode=1).build()
    model = DQN(
        CNNFeaturesPolicy,
        env,
        verbose=1,
        policy_kwargs=dict(
           features_extractor_kwargs=dict(
                ac_space=env.action_space,
                mask_function=None
            )
        )
    )