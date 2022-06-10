import datetime
import logging
from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from power_perceiver.consts import BatchKey
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    SpatialDataSource,
    TimeseriesDataSource,
    ZarrDataSource,
)
from power_perceiver.time import date_summary_str
from power_perceiver.utils import datetime64_to_float, is_sorted

_log = logging.getLogger(__name__)


# Means and std computed with
# nowcasting_dataset/scripts/compute_stats_from_batches.py
# using v15 training batches on 2021-11-24.
NWP_MEAN = {
    "t": 285.7799539185846,
    "dswrf": 294.6696933986283,
    "prate": 3.6078121378638696e-05,
    "r": 75.57106712435926,
    "sde": 0.0024915961594965614,
    "si10": 4.931356852411006,
    "vis": 22321.762918384553,
    "lcc": 47.90454236572895,
    "mcc": 44.22781694449808,
    "hcc": 32.87577371914454,
}

NWP_STD = {
    "t": 5.017000766747606,
    "dswrf": 233.1834250473355,
    "prate": 0.00021690701537950742,
    "r": 15.705370079694358,
    "sde": 0.07560040052148084,
    "si10": 2.664583614352396,
    "vis": 12963.802514945439,
    "lcc": 40.06675870700349,
    "mcc": 41.927221148316384,
    "hcc": 39.05157559763763,
}

NWP_CHANNEL_NAMES = list(NWP_STD.keys())


def _to_data_array(d):
    return xr.DataArray(
        [d[key] for key in NWP_CHANNEL_NAMES], coords={"channel": NWP_CHANNEL_NAMES}
    )


NWP_MEAN = _to_data_array(NWP_MEAN)
NWP_STD = _to_data_array(NWP_STD)


@dataclass(kw_only=True)
class RawNWPDataSource(
    # Surprisingly, Python's class hierarchy is defined right-to-left.
    # So the base class must go on the right.
    ZarrDataSource,
    TimeseriesDataSource,
    SpatialDataSource,
    RawDataSource,
):
    """Load NWPs direction from the intermediate NWP Zarr store.

    In particular, loads the "UKV" NWP from the UK Met Office, downloaded from CEDA.
    "UKV" stands for "United Kingdom Variable", and it the UK Met Office's high-res deterministic
    NWP for the UK. All the NWP variables are represented in the `variable` dimension within
    the UKV DataArray.

    x is left-to-right.
    y is top-to-bottom.

    Init args:
        zarr_path:
        roi_height_pixels:
        roi_width_pixels:
        history_duration: datetime.timedelta
        forecast_duration: datetime.timedelta
        start_date: datetime.datetime
        end_date: datetime.datetime
        y_coarsen:
        x_coarsen: Downsample by taking the mean across this number of pixels.
        channels: The NWP forecast parameters to load. If None then don't filter.
            See:  http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf
            All of these params are "instant" (i.e. a snapshot at the target time,
            not accumulated over some time period).  The available params are:
                cdcb  : Height of lowest cloud base > 3 oktas, in meters above surface.
                lcc   : Low-level cloud cover in %.
                mcc   : Medium-level cloud cover in %.
                hcc   : High-level cloud cover in %.
                sde   : Snow depth in meters.
                hcct  : Height of convective cloud top, meters above surface.
                        WARNING: hcct has NaNs where there are no clouds forecast to exist!
                dswrf : Downward short-wave radiation flux in W/m^2 (irradiance) at surface.
                dlwrf : Downward long-wave radiation flux in W/m^2 (irradiance) at surface.
                h     : Geometrical height, meters.
                t     : Air temperature at 1 meter above surface in Kelvin.
                r     : Relative humidty in %.
                dpt   : Dew point temperature in Kelvin.
                vis   : Visibility in meters.
                si10  : Wind speed in meters per second, 10 meters above surface.
                wdir10: Wind direction in degrees, 10 meters above surface.
                prmsl : Pressure reduce to mean sea level in Pascals.
                prate : Precipitation rate at the surface in kg/m^2/s.
    """

    y_coarsen: int
    x_coarsen: int
    channels: Optional[Sequence[str]] = None

    # Attributes which are intended to be set for the whole class:
    sample_period_duration: ClassVar[datetime.timedelta] = datetime.timedelta(hours=1)
    needs_to_load_subset_into_ram: ClassVar[bool] = True
    _y_dim_name: ClassVar[str] = "y_osgb"
    _x_dim_name: ClassVar[str] = "x_osgb"
    _time_dim_name: ClassVar[str] = "init_time_utc"

    def __post_init__(self):  # noqa: D105
        RawDataSource.__post_init__(self)
        SpatialDataSource.__post_init__(self)
        TimeseriesDataSource.__post_init__(self)
        ZarrDataSource.__post_init__(self)

    def per_worker_init(self, worker_id: int) -> None:  # noqa: D102
        super().per_worker_init(worker_id)
        self.open()

    def open(self) -> None:
        """
        Lazily open NWP data (this only loads metadata. It doesn't load all the data!)

        We don't want to call `open_sat_data` in __init__.
        If we did that, then we couldn't copy RawNWPDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        self._data_on_disk = open_nwp(zarr_path=self.zarr_path)
        if self.channels is not None:
            self._data_on_disk = self._data_on_disk.sel(channel=self.channels)
        self.channels = self.data_on_disk.channel.values

        # Check the x and y coords are sorted.
        assert is_sorted(self.data_on_disk[self._y_dim_name][::-1])
        assert is_sorted(self.data_on_disk[self._x_dim_name])

        # Sub-select data:
        _log.info("Before any selection: " + date_summary_str(self.data_on_disk.init_time_utc))

        # Select only the timesteps we want:
        self._data_on_disk = self.data_on_disk.sel(
            init_time_utc=slice(self.start_date, self.end_date)
        )
        # Downsample spatially:
        self._data_on_disk = self.data_on_disk.coarsen(
            y_osgb=self.y_coarsen,
            x_osgb=self.x_coarsen,
            boundary="trim",
        ).mean()
        _log.info("After filtering: " + date_summary_str(self.data_on_disk.init_time_utc))

    @property
    def datetime_index(self) -> pd.DatetimeIndex:
        """Return a complete list of all available datetimes.

        We need to return the `target_time_utc` (the times the NWPs are _about_).
        The `target_time` is the `init_time_utc` plus the forecast horizon `step`.
        `step` is an array of timedeltas, so we can just add `init_time_utc` to `step`.
        """
        target_times = self.data_on_disk.init_time_utc + self.data_on_disk.step
        target_times = target_times.values.flatten()
        target_times = np.unique(target_times)
        target_times = np.sort(target_times)
        target_times = pd.DatetimeIndex(target_times)
        return target_times

    def _get_time_slice(
        self, xr_data: xr.DataArray, t0_datetime_utc: datetime.datetime
    ) -> xr.DataArray:
        """Select a timeslice from `xr_data`.

        The returned data does not include an `example` dimension.
        """
        start_dt_ceil = self._get_start_dt_ceil(t0_datetime_utc)
        end_dt_ceil = self._get_end_dt_ceil(t0_datetime_utc)

        xr_data_at_one_init_time = xr_data.sel(init_time_utc=start_dt_ceil, method="pad")
        most_recent_init_time = xr_data_at_one_init_time.init_time_utc.values

        start_step = start_dt_ceil - most_recent_init_time
        end_step = end_dt_ceil - most_recent_init_time

        # Get time slice:
        time_slice = xr_data_at_one_init_time.sel(step=slice(start_step, end_step))
        time_slice = time_slice.swap_dims({"step": "target_time_utc"})
        time_slice["target_time_utc"] = most_recent_init_time + time_slice.step

        self._sanity_check_time_slice(time_slice, "target_time_utc", t0_datetime_utc)
        return time_slice

    def _post_process(self, xr_data: xr.DataArray) -> xr.DataArray:
        xr_data = xr_data - NWP_MEAN
        xr_data = xr_data / NWP_STD
        return xr_data

    @staticmethod
    def to_numpy(xr_data: xr.DataArray) -> NumpyBatch:
        """Convert xarray to numpy batch.

        But note that this is actually just returns *one* example (not a whole batch!)
        """
        example: NumpyBatch = {}

        example[BatchKey.nwp] = xr_data.values
        example[BatchKey.nwp_init_time_utc] = datetime64_to_float(xr_data.init_time_utc.values)
        example[BatchKey.nwp_target_time_utc] = datetime64_to_float(xr_data.target_time_utc.values)

        for batch_key, dataset_key in (
            (BatchKey.nwp_y_osgb, "y_osgb"),
            (BatchKey.nwp_x_osgb, "x_osgb"),
        ):
            example[batch_key] = xr_data[dataset_key].values

        return example


def open_nwp(zarr_path: str) -> xr.DataArray:
    """
    Open The NWP data.

    Args:
        zarr_path: zarr_path must start with 'gs://' if it's on GCP.

    Returns: NWP data.
    """
    _log.debug("Opening NWP data: %s", zarr_path)
    nwp = xr.open_dataset(
        zarr_path,
        engine="zarr",
        consolidated=True,
        mode="r",
        chunks="auto",
    )

    ukv = nwp["UKV"]
    del nwp

    ukv = ukv.transpose("init_time", "step", "variable", "y", "x")
    ukv = ukv.rename(
        {"init_time": "init_time_utc", "variable": "channel", "y": "y_osgb", "x": "x_osgb"}
    )

    # y_osgb and x_osgb are int64 on disk.
    for coord_name in ("y_osgb", "x_osgb"):
        ukv[coord_name] = ukv[coord_name].astype(np.float32)

    # Sanity checks.
    assert ukv.y_osgb[0] > ukv.y_osgb[1], "UKV must run from top-to-bottom."
    time = pd.DatetimeIndex(ukv.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing

    return ukv
