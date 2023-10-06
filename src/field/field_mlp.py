import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from src.components.positional_encoding import PositionalEncoding

from .field import Field


class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)
        layers = []
        input_dim = d_coordinate

        # positional encoding
        if cfg.positional_encoding_octaves is not None:
            self.positional_encoding = PositionalEncoding(
                cfg.positional_encoding_octaves
            )
            input_dim *= cfg.positional_encoding_octaves * 2
        else:
            self.positional_encoding = None

        # Create hidden layers
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(input_dim, cfg.d_hidden))
            layers.append(nn.ReLU())
            input_dim = cfg.d_hidden

        # Output layer
        layers.append(nn.Linear(input_dim, d_out))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""
        if self.positional_encoding:
            coordinates = self.positional_encoding(coordinates)
            # Reshape or flatten the positional encoding dimensions
        coordinates = coordinates.clone().detach().requires_grad_(True)
        output = self.mlp(coordinates)
        return output
