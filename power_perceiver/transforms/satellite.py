"""Satellite transforms."""

from dataclasses import dataclass

import xarray as xr

from power_perceiver.data_loader.satellite import _set_sat_coords


@dataclass
class PatchSatellite:
    """Appends a `patch` dimension to the satellite data.

    The `patch` dim will be x_patch_size_pixels * y_patch_size_pixels in size.
    """

    y_patch_size_pixels: int = 4
    x_patch_size_pixels: int = 4

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset = dataset.coarsen(y=self.y_patch_size_pixels, x=self.x_patch_size_pixels)
        dataset = dataset.construct(y=("y", "y_patch"), x=("x", "x_patch"))
        dataset = dataset.stack(patch=("y_patch", "x_patch"))

        # Downsample spatial coordinates
        COORD_NAMES = ["x_geostationary", "x_osgb", "y_geostationary", "y_osgb"]
        dataset[COORD_NAMES] = dataset[COORD_NAMES].mean(dim="patch")

        dataset = _set_sat_coords(dataset)

        return dataset
