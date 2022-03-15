from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from power_perceiver.consts import DataSourceName, XarrayBatch


@dataclass
class Select1PVSystem:
    """Selects a single PV system near the top-middle of the HRV satellite imagery.

    Initialisation args:
        encode_pixel_positions_relative_to_pv_system: If True then append two new items
            to the xr_batch dict, with keys:
                angle_in_degrees_from_pv_system
                distance_in_meters_from_pv_system
        image_data_source_name: The name of the data source which defines the geospatial
            boundaries we'll use to select PV systems.
        geo_border_km: When selecting a single PV system from the top-middle of the HRV satellite
            imagery, use geo_border_km to define how much to reduce the selection rectangle
            before selecting PV systems. In kilometers.

    Attributes:
        rng: A numpy random number generator.
    """

    encode_pixel_positions_relative_to_pv_system: bool = True
    image_data_source_name: DataSourceName = DataSourceName.hrvsatellite
    geo_border_km: pd.Series = pd.Series(dict(left=10, right=10, bottom=57, top=16))

    def __post_init__(self):
        self.rng = np.random.default_rng(seed=42)

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        xr_batch = self._select_1_pv_system(xr_batch)
        if self.encode_pixel_positions_relative_to_pv_system:
            xr_batch = _encode_pixel_positions_relative_to_pv_system(xr_batch)
        return xr_batch

    def _select_1_pv_system(self, xr_batch: XarrayBatch) -> XarrayBatch:
        image_dataset = xr_batch[self.image_data_source_name]
        pv_dataset = xr_batch[DataSourceName.pv]
        batch_size = image_dataset.dims["example"]
        pv_id_indexes_for_all_examples = []
        # If this loop is too slow then it may be possible to vectorise this code, to work on
        # all examples at once.
        for example_i in range(batch_size):
            # Get inner rectangle for image dataset:
            image_dataset_for_example = image_dataset.sel(example=example_i)
            inner_rectangle = self._get_maximal_regular_inner_rectangle(image_dataset_for_example)

            # Reduce inner rectangle:
            inner_rectangle["left"] += self.geo_border_km["left"]
            inner_rectangle["right"] -= self.geo_border_km["right"]
            inner_rectangle["bottom"] += self.geo_border_km["bottom"]
            inner_rectangle["top"] -= self.geo_border_km["top"]

            # Find PV systems within the inner rectangle:
            pv_dataset_for_example = pv_dataset.sel(example=example_i)
            pv_locations = pv_dataset_for_example[["x_coords", "y_coords"]]
            valid_pv_id_mask = np.isfinite(pv_dataset_for_example.id.values)
            pv_locations = pv_locations.isel(id_index=valid_pv_id_mask)
            pv_system_selection_mask = (
                (pv_locations.x_coords >= inner_rectangle["left"])
                & (pv_locations.x_coords <= inner_rectangle["right"])
                & (pv_locations.y_coords >= inner_rectangle["bottom"])
                & (pv_locations.y_coords <= inner_rectangle["top"])
            )
            selected_pv_id_indexes = pv_locations.id_index[pv_system_selection_mask]
            assert len(selected_pv_id_indexes) > 0, "No PV systems in selection mask!"

            # Select 1 PV system:
            pv_id_index = self.rng.choice(selected_pv_id_indexes)
            pv_id_indexes_for_all_examples.append([pv_id_index])

        # select one PV ID index from each example:
        id_indexes = xr.DataArray(
            pv_id_indexes_for_all_examples,
            dims=["example", "id_index"],
            coords={"example": pv_dataset.example},
        )
        pv_dataset = pv_dataset.sel(id_index=id_indexes)
        xr_batch[DataSourceName.pv] = pv_dataset
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
        bounds = pd.Series(index=["left", "right", "top", "bottom"])

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


def _encode_pixel_positions_relative_to_pv_system(xr_batch: XarrayBatch) -> XarrayBatch:
    # TODO
    return xr_batch