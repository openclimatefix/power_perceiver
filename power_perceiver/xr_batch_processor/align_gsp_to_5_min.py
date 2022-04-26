from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from power_perceiver.consts import BatchKey
from power_perceiver.data_loader import GSP, DataLoader, HRVSatellite, XarrayBatch
from power_perceiver.data_loader.data_loader import NumpyBatch
from power_perceiver.utils import datetime64_to_int


class GSP5Min(DataLoader):
    """This doesn't actually 'load' data :)

    Instead it's a hack to allow us to insert `gsp_5_min` stuff into the NumpyBatch."""

    @staticmethod
    def to_numpy(dataset: xr.Dataset) -> NumpyBatch:
        """This is called from Dataset.__getitem__.

        This processes this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        batch: NumpyBatch = {}
        batch[BatchKey.gsp_5_min] = dataset["power_normalised"].values
        batch[BatchKey.gsp_5_min_time_utc] = datetime64_to_int(dataset["time_utc"].values)
        return batch


@dataclass
class AlignGSPTo5Min:
    """Adds an entry to the XarrayBatch where the GSP data is aligned to the 5 min data.

    The GSP data isn't interpolated. Instead, for each 5_min_timestep, we
    take the GSP data at 5_min_timestep.ceil("30T"). If that GSP timestep does not
    exist then NaNs will be used.

    Specifically, adds an item whose key is the `GSP5Min` class, and
    the value is the xr.Dataset with 5 minutely GSP data.
    """

    data_loader_class_for_5_min: DataLoader = HRVSatellite

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        gsp_dataset = xr_batch[GSP]
        five_min_dataset = xr_batch[self.data_loader_class_for_5_min]

        # Find the corresponding GSP 30 minute timestep for each 5 minute satellite timestep.
        # We do this by taking the `ceil("30T")` of each 5 minute satellite timestep.
        # Most of the code below is just converting from xarray to Pandas and back
        # so we can use `pd.DatetimeIndex.ceil` on each datetime:
        time_5_min_series = five_min_dataset.time_utc.to_series()
        time_5_min_dt_index = pd.DatetimeIndex(time_5_min_series)
        time_30_min_dt_index = time_5_min_dt_index.ceil("30T")
        time_30_min_series = pd.Series(time_30_min_dt_index, index=time_5_min_series.index)
        time_30_min_da = time_30_min_series.to_xarray()

        # Loop through each example and find the index into the GSP time dimension
        # of the GSP timestep corresponding to each 5 minute satellite timestep:
        gsp_5_min_for_all_examples = []
        max_time_idx = len(gsp_dataset.time) - 1
        for example_i in gsp_dataset.example:
            idx_into_gsp = np.searchsorted(
                gsp_dataset.sel(example=example_i).time_utc.values,
                time_30_min_da.sel(example=example_i).values,
            )
            gsp_5_min = gsp_dataset.isel(
                example=example_i, time=idx_into_gsp.clip(max=max_time_idx)
            )

            # Now, for any timestep where we don't have GSP data, set to NaN:
            mask = idx_into_gsp <= max_time_idx
            gsp_5_min = gsp_5_min.where(mask)
            gsp_5_min["time_utc"] = gsp_5_min.time_utc.where(mask)
            gsp_5_min_for_all_examples.append(gsp_5_min)

        xr_batch[GSP5Min] = xr.concat(gsp_5_min_for_all_examples, dim="example")
        return xr_batch
