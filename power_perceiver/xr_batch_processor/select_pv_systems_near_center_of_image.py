import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from power_perceiver.data_loader import PV, DataLoader, HRVSatellite, XarrayBatch

_log = logging.getLogger(__name__)


@dataclass
class SelectPVSystemsNearCenterOfImage:
    """Selects PV systems near the top-middle of the imagery.

    The returned XarrayBatch will have the same number of spaces for PV systems as the input
    XarrayBatch, but the PV systems which are unselected will be set to NaN.

    Initialisation args:
        image_data_loader_class: The name of the data source which defines the geospatial
            boundaries we'll use to select PV systems.
        geo_border_km: When selecting a single PV system from the top-middle of the HRV satellite
            imagery, use geo_border_km to define how much to reduce the selection rectangle
            before selecting PV systems. In kilometers.
        drop_examples: If True then drop examples (from all data sources) which have no PV systems.

    Attributes:
        rng: A numpy random number generator.
    """

    image_data_loader_class: DataLoader = HRVSatellite
    geo_border_km: pd.Series = pd.Series(dict(left=8, right=8, bottom=32, top=16))
    drop_examples: bool = True

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        image_dataset = xr_batch[self.image_data_loader_class]
        pv_dataset = xr_batch[PV]
        batch_size = image_dataset.dims["example"]
        pv_id_indexes_for_all_examples = []
        # If this loop is too slow then it may be possible to vectorise this code.
        for example_i in range(batch_size):
            # Get inner rectangle for image dataset:
            image_dataset_for_example = image_dataset.sel(example=example_i)
            inner_rectangle = self._get_maximal_regular_inner_rectangle(image_dataset_for_example)

            # Reduce inner rectangle:
            inner_rectangle["left"] += self.geo_border_km["left"]
            inner_rectangle["right"] -= self.geo_border_km["right"]
            inner_rectangle["bottom"] += self.geo_border_km["bottom"]
            inner_rectangle["top"] -= self.geo_border_km["top"]

            # Sanity check:
            if any(inner_rectangle.isnull()):
                msg = f"At least one inner_rectangle value is NaN!\n{inner_rectangle=}"
                _log.error(msg)
                raise ValueError(msg)

            # Find PV systems within the inner rectangle:
            pv_dataset_for_example = pv_dataset.sel(example=example_i)
            pv_locations = pv_dataset_for_example[["x_coords", "y_coords"]]
            pv_system_selection_mask = (
                (pv_locations.x_coords >= inner_rectangle["left"])
                & (pv_locations.x_coords <= inner_rectangle["right"])
                & (pv_locations.y_coords >= inner_rectangle["bottom"])
                & (pv_locations.y_coords <= inner_rectangle["top"])
            )
            pv_id_indexes_for_all_examples.append(pv_system_selection_mask)

        # Set PV systems outside of the inner_rectangle to NaN.
        mask = xr.concat(pv_id_indexes_for_all_examples, dim="example")
        xr_batch[PV] = pv_dataset.where(mask)

        if self.drop_examples:
            # Drop examples which don't have any PV systems.
            n_pv_systems_per_example = mask.sum(dim="id_index")
            examples_to_drop = np.where(n_pv_systems_per_example == 0)[0]
            for data_loader_class, xr_dataset in xr_batch.items():
                xr_batch[data_loader_class] = xr_dataset.drop_sel(example=examples_to_drop)

        return xr_batch

    def _get_maximal_regular_inner_rectangle(self, image_dataset: xr.Dataset) -> pd.Series:
        """Find the geographical boundaries (in OSGB coordinates) of the regular inner rectangle.

        Getting the OSGB boundaries for the PV systems is a bit fiddly because the satellite image
        is a skewed rectangle. We want the regular rectangle that fits within the OSGB boundaries:

            skewed satellite image                # noqa W605
            \--------\
             \         \-------\
              \    |===========|\
               \   | selection | \
                \  |           |  \
                 \ |===========|   \
                  \--------\        \
                            \--------\

        And the skew angle and direction might vary (e.g. if we compare images from Italy vs UK).
        So, to find the left value for _x_, we find the left-most coordinate for the top and
        bottom rows satellite image, and find the _max_ (right-most) value of this pair of numbers.

        Args:
            image_dataset: xarray Dataset for a single example. Must have these coordinates:
                y_geostationary_index, x_geostationary_index, y_osgb, x_osgb

        Returns:
            pd.Series with keys left, right, top, bottom. The values are the OSGB coordinates of the relevant axis.
              e.g. the left and right are the values for the x axis.
        """
        bounds = pd.Series(index=["left", "right", "top", "bottom"], dtype=np.float32)

        # Handle X coordinates. Get the x_osgb coords for the top and bottom rows of pixels.
        x_osgb = image_dataset["x_osgb"].isel(y_geostationary_index=[0, -1])
        bounds["left"] = x_osgb.isel(x_geostationary_index=0).max().values
        bounds["right"] = x_osgb.isel(x_geostationary_index=-1).min().values

        # Handle Y coordinates. Get the y_osgb coords for the left and right columns of pixels.
        y_osgb = image_dataset["y_osgb"].isel(x_geostationary_index=[0, -1])
        bounds["bottom"] = y_osgb.isel(y_geostationary_index=0).max().values
        bounds["top"] = y_osgb.isel(y_geostationary_index=-1).min().values

        # Sanity check!
        x_osgb_range = bounds["right"] - bounds["left"]
        y_osgb_range = bounds["top"] - bounds["bottom"]
        assert x_osgb_range > 0
        assert y_osgb_range > 0

        return bounds
