import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import DQN

class CNNFeaturesPolicy(BasePolicy):
    def __init__(self, ob_space, ac_space, net_arch, features_extractor=None, **kwargs):
        super(CNNFeaturesPolicy, self).__init__()

        # Capas convolucionales para procesar la imagen
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Capas totalmente conectadas para el estado auxiliar y la salida
        self.fc_aux = nn.Linear(2, 256)
        self.fc_combined = nn.Linear(64 * 7 * 7 + 256, ac_space.n)

        # Activación ReLU
        self.relu = nn.ReLU()

    def forward(self, obs, state=None, mask=None):
        # Divide las observaciones en las partes correspondientes
        image_obs = obs['image_state']
        aux_obs = obs['aux_state']

        # Procesa la imagen a través de las capas convolucionales
        x = self.relu(self.conv1(image_obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Aplana la salida de las capas convolucionales

        # Procesa el estado auxiliar a través de la capa totalmente conectada
        aux_x = self.relu(self.fc_aux(aux_obs))

        # Concatena las salidas de las capas convolucionales y el estado auxiliar
        combined_features = torch.cat((x, aux_x), dim=1)

        # Procesa la salida a través de la capa totalmente conectada final
        output = self.fc_combined(combined_features)

        # TODO apply masks

        return output, None
