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
                torch.randn(1, d_out, self.side_length, self.side_length)
            )
        elif d_coordinate == 3:
            self.grid = torch.nn.Parameter(
                torch.randn(
                    1,
                    d_out,
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
        coordinates = (coordinates / (self.side_length - 1)) * 2 - 1
        coordinates = coordinates.unsqueeze(0).unsqueeze(2)  # 1 batch 1 d_coordinate
        # Use grid_sample
        sampled_values = torch.nn.functional.grid_sample(
            input=self.grid,
            grid=coordinates,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_values = (
            sampled_values.squeeze(0).squeeze(-1).permute(1, 0)
        )  # [batch, d_out]

        return sampled_values
