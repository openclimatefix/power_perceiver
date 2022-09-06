import datetime
import logging
from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    SpatialDataSource,
    TimeseriesDataSource,
    ZarrDataSource,
)
from power_perceiver.time import date_summary_str
from power_perceiver.utils import datetime64_to_float, is_sorted, select_time_periods

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

NWP_CHANNEL_NAMES = tuple(NWP_STD.keys())


def _to_data_array(d):
    return xr.DataArray(
        [d[key] for key in NWP_CHANNEL_NAMES], coords={"channel": list(NWP_CHANNEL_NAMES)}
    ).astype(np.float32)


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
        time_periods: pd.DataFrame
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
    duration_between_nwp_inits: ClassVar[datetime.timedelta] = datetime.timedelta(hours=3)
    needs_to_load_subset_into_ram: ClassVar[bool] = True
    _y_dim_name: ClassVar[str] = "y_osgb"
    _x_dim_name: ClassVar[str] = "x_osgb"
    _time_dim_name: ClassVar[str] = "init_time_utc"

    def __post_init__(self):  # noqa: D105
        if self.channels:
            chans_not_in_channel_names = set(self.channels) - set(NWP_CHANNEL_NAMES)
            assert len(chans_not_in_channel_names) == 0, (
                f"{len(chans_not_in_channel_names)} requested channel name(s) not in"
                f" NWP_CHANNEL_NAMES! {chans_not_in_channel_names=}; {self.channels=};"
                f" {NWP_CHANNEL_NAMES=}"
            )
        RawDataSource.__post_init__(self)
        SpatialDataSource.__post_init__(self)
        TimeseriesDataSource.__post_init__(self)
        ZarrDataSource.__post_init__(self)
        self.empty_example = self._get_empty_example()

    def _get_empty_example(self) -> xr.DataArray:
        target_time_utc = pd.DatetimeIndex([np.NaN] * self.total_seq_length)
        channels = self.channels if self.channels else NWP_CHANNEL_NAMES
        n_channels = len(channels)
        data = np.full(
            shape=(
                self.total_seq_length,
                n_channels,
                self.roi_height_pixels,
                self.roi_width_pixels,
            ),
            fill_value=np.NaN,
            dtype=np.float32,
        )
        y = np.full(
            shape=self.roi_height_pixels,
            fill_value=np.NaN,
            dtype=np.float32,
        )
        x = np.full(
            shape=self.roi_width_pixels,
            fill_value=np.NaN,
            dtype=np.float32,
        )
        data_array = xr.DataArray(
            data,
            coords=(
                ("target_time_utc", target_time_utc),
                ("channel", list(channels)),
                ("y_osgb", y),
                ("x_osgb", x),
            ),
        )
        data_array.attrs["t0_idx"] = self.t0_idx
        return data_array

    def per_worker_init(self, *args, **kwargs) -> None:  # noqa: D102
        super().per_worker_init(*args, **kwargs)
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
            self._data_on_disk = self._data_on_disk.sel(channel=list(self.channels))
        self.channels = self.data_on_disk.channel.values

        # Check the x and y coords are sorted.
        assert is_sorted(self.data_on_disk[self._y_dim_name][::-1])
        assert is_sorted(self.data_on_disk[self._x_dim_name])

        # Sub-select data:
        _log.info("Before any selection: " + date_summary_str(self.data_on_disk.init_time_utc))

        # Select only the timesteps we want:
        self._data_on_disk = select_time_periods(
            xr_data=self.data_on_disk,
            time_periods=self.time_periods,
            dim_name=self._time_dim_name,
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
        # Only include steps that are at most `total_duration` apart.
        data = self.data_on_disk.sel(step=slice(None, self.total_duration))
        target_times = data.init_time_utc + data.step
        target_times = target_times.values.flatten()
        target_times = np.unique(target_times)
        target_times = np.sort(target_times)
        target_times = pd.DatetimeIndex(target_times)
        return target_times

    def _get_time_slice(
        self, xr_data: xr.DataArray, t0_datetime_utc: datetime.datetime
    ) -> xr.DataArray:
        """Select a timeslice from `xr_data`.

        The NWP for each example covers a contiguous timespan running from `start_dt` to `end_dt`.
        The first part of the timeseries [`start_dt`, `t0`] is the 'recent history'.  The second
        part of the timeseries (`t0`, `end_dt`] is the 'future'. For each timestep in the
        recent history [`start`, `t0`], get predictions produced by the freshest NWP initialisation
        to each target_time_utc. For the future (`t0`, `end`], use the NWP initialised most
        recently to t0.

        The returned data does not include an `example` dimension.
        """
        start_dt_ceil = self._get_start_dt_ceil(t0_datetime_utc)
        end_dt_ceil = self._get_end_dt_ceil(t0_datetime_utc)

        target_times = pd.date_range(start_dt_ceil, end_dt_ceil, freq=self.sample_period_duration)

        # Get the most recent NWP initialisation time for each target_time_hourly.
        init_times = xr_data.sel(init_time_utc=target_times, method="pad").init_time_utc.values

        # Find the NWP init time for just the 'future' portion of the example.
        init_time_t0 = init_times[self.t0_idx]

        # For the 'future' portion of the example, replace all the NWP
        # init times with the NWP init time most recent to t0.
        init_times[self.t0_idx :] = init_time_t0

        steps = target_times - init_times

        # We want one timestep for each target_time_hourly (obviously!) If we simply do
        # nwp.sel(init_time=init_times, step=steps) then we'll get the *product* of
        # init_times and steps, which is not what # we want! Instead, we use xarray's
        # vectorized-indexing mode by using a DataArray indexer.  See the last example here:
        # https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing
        coords = {"target_time_utc": target_times}
        init_time_indexer = xr.DataArray(init_times, coords=coords)
        step_indexer = xr.DataArray(steps, coords=coords)
        time_slice = xr_data.sel(step=step_indexer, init_time_utc=init_time_indexer)

        self._sanity_check_time_slice(time_slice, "target_time_utc", t0_datetime_utc)
        return time_slice

    def _post_process(self, xr_data: xr.DataArray) -> xr.DataArray:
        xr_data = xr_data - NWP_MEAN
        xr_data = xr_data / NWP_STD
        return xr_data

    def _convert_t0_time_periods_to_periods_to_load(
        self, subset_of_contiguous_t0_time_periods: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert t0_time_periods back into the complete time periods we want to load.

        Need to stretch the `time_periods` because the nwp init_time is only at
        `duration_between_nwp_inits` intervals.
        """
        time_periods = super()._convert_t0_time_periods_to_periods_to_load(
            subset_of_contiguous_t0_time_periods
        )
        # Get the init time that's most recent to each start_dt.
        time_periods["start_dt"] = self.data_on_disk.sel(
            init_time_utc=time_periods.start_dt.values, method="pad"
        ).init_time_utc
        return time_periods

    @staticmethod
    def to_numpy(xr_data: xr.DataArray) -> NumpyBatch:
        """Convert xarray to numpy batch.

        But note that this is actually just returns *one* example (not a whole batch!)
        """
        example: NumpyBatch = {}

        example[BatchKey.nwp] = xr_data.values
        example[BatchKey.nwp_t0_idx] = xr_data.attrs["t0_idx"]
        target_time = xr_data.target_time_utc.values
        example[BatchKey.nwp_target_time_utc] = datetime64_to_float(target_time)
        example[BatchKey.nwp_channel_names] = xr_data.channel.values
        example[BatchKey.nwp_step] = (xr_data.step.values / np.timedelta64(1, "h")).astype(np.int64)
        example[BatchKey.nwp_init_time_utc] = datetime64_to_float(xr_data.init_time_utc.values)

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
