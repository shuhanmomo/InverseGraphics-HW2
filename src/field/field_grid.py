import torch
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field


class FieldGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a grid for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/grid.yaml):

        - side_length: the side length in each dimension

        Your architecture only needs to support 2D and 3D grids.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate in (2, 3)
        self.side_length = cfg.side_length
        self.d_coordinate = d_coordinate
        self.d_out = d_out

        # Initializing a learnable tensor for the grid
        if d_coordinate == 2:
            self.grid = torch.nn.Parameter(
                torch.randn(d_out, self.side_length, self.side_length)
            )
        elif d_coordinate == 3:
            self.grid = torch.nn.Parameter(
                torch.randn(
                    1,
                    self.side_length,
                    self.side_length,
                    self.side_length,
                )
            )

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """
        # normalization
        batch = coordinates.size(0)
        coordinates = coordinates * 2 - 1
        if self.d_coordinate == 2:
            coordinates = coordinates.view(1, 1, batch, 2)
        else:
            coordinates = coordinates.view(1, 1, 1, batch, 3)
        # Use grid_sample
        sampled_values = torch.nn.functional.grid_sample(
            input=self.grid.unsqueeze(0),
            grid=coordinates,
            mode="bilinear",
            align_corners=False,
        )
        sampled_values = (
            sampled_values.transpose(1, -2).transpose(-1, -2).reshape(batch, self.d_out)
        )
        # [batch, d_out]
        return sampled_values
