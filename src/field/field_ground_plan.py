import torch
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from src.components.positional_encoding import PositionalEncoding

from .field import Field
from .field_grid import FieldGrid
from .field_mlp import FieldMLP


class FieldGroundPlan(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a neural ground plan. You should reuse the following components:

        - FieldGrid from  src/field/field_grid.py
        - FieldMLP from src/field/field_mlp.py
        - PositionalEncoding from src/components/positional_encoding.py

        Your ground plan only has to handle the 3D case.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate == 3
        self.field_grid = FieldGrid(cfg.grid, 2, d_out)
        self.positional_encoding = PositionalEncoding(cfg.positional_encoding_octaves)
        mlp_input_dim = d_out + self.positional_encoding.d_out(1)
        self.field_mlp = FieldMLP(cfg.mlp, mlp_input_dim, d_out)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the ground plan at the specified coordinates. You should:

        - Sample the grid using the X and Y coordinates.
        - Positionally encode the Z coordinates.
        - Concatenate the grid's outputs with the corresponding encoded Z values, then
          feed the result through the MLP.
        """
        # sample grid using x y coord
        xy_coords = coordinates[:, :2]
        grid_out = self.field_grid(xy_coords)

        # positional encoding of z
        z_coords = coordinates[:, -1].unsqueeze(-1)
        encoded_z = self.positional_encoding(z_coords)

        # concatenation
        concatenated_out = torch.cat([grid_out, encoded_z], dim=-1)

        # MLP
        mlp_out = self.field_mlp(concatenated_out)

        return mlp_out
