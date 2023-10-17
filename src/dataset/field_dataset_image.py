import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

from .field_dataset import FieldDataset


class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        super().__init__(cfg)
        self.cfg = cfg
        read_image = Image.open(cfg.path).convert("RGB")

        self.image = (
            torch.tensor(np.array(read_image)).permute(2, 0, 1).unsqueeze(0).float()
            / 255
        )  # batch channel h w

    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """
        self.image = self.image.to(coordinates.device)
        coordinates = coordinates * 2 - 1
        coordinates = coordinates.unsqueeze(0).unsqueeze(2)
        print(f"self.image shape is {self.image.shape}")
        print(f"coordinates shape is {coordinates.shape}")
        sampled_colors = F.grid_sample(
            self.image, coordinates, align_corners=False
        )  # batch channel d_out 1
        print(f"sampled color after grid sample is {sampled_colors.shape}")
        output = sampled_colors.squeeze(0).squeeze(-1).permute(1, 0)
        print(f"output after permutation is {output.shape}")
        return output  # batch d_out

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""
        return self.image.shape[-2:]
