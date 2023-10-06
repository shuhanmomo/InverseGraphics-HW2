import numpy as np
import torch
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from src.components.sine_layer import SineLayer

from .field import Field


class FieldSiren(Field):
    network: nn.Sequential

    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a SIREN network using the sine layers at src/components/sine_layer.py.
        Your network should consist of:

        - An input sine layer whose output dimensionality is 256
        - Two hidden sine layers with width 256
        - An output linear layer
        """
        super().__init__(cfg, d_coordinate, d_out)
        layers = []
        input_dim = d_coordinate
        hidden_dim = 256
        hidden_num = 2
        hidden_omega_0 = 30.0

        # append first layer
        layers.append(SineLayer(input_dim, hidden_dim, is_first=True))

        # append hidden layers
        for _ in range(hidden_num):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False))

        # append output layer
        output_layer = nn.Linear(hidden_dim, d_out)
        with torch.no_grad():
            output_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / hidden_omega_0,
                np.sqrt(6 / hidden_dim) / hidden_omega_0,
            )
        layers.append(output_layer)

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        coordinates = coordinates.clone().detach().requires_grad_(True)
        output = self.mlp(coordinates)
        return output
