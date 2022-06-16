import datetime
import logging
from dataclasses import dataclass

import pandas as pd
import pvlib
import xarray as xr

from power_perceiver.consts import BatchKey, Location
from power_perceiver.geospatial import osgb_to_lat_lon
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_prepared_batches.data_sources.sun import (
    AZIMUTH_MEAN,
    AZIMUTH_STD,
    ELEVATION_MEAN,
    ELEVATION_STD,
)
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    TimeseriesDataSource,
)

_log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RawSunPositionDataSource(
    # Surprisingly, Python's class hierarchy is defined right-to-left.
    # So the base class must go on the right.
    TimeseriesDataSource,
    RawDataSource,
):
    """
    Data source for the Sun's azimuth and elevation.

    This is a duplicate of the info in the Sun pre-prepared batch.

    But we don't have access to those pre-prepared batches when training directly
    from the Zarr! Hence we need this when training directly from Zarr!

    COMPLETELY UNTESTED!
    """

    # Deliberately change sample_period_duration to an instance variable (instead of
    # a class variable) because the user might want to change this.
    sample_period_duration: datetime.timedelta = datetime.timedelta(minutes=5)

    def __post_init__(self):  # noqa: D105
        RawDataSource.__post_init__(self)
        TimeseriesDataSource.__post_init__(self)

    def get_osgb_location_for_example(self) -> Location:
        """Get a single random geographical location."""
        raise NotImplementedError("RawSunPositionDataSource cannot be used to select locations!")

    @property
    def empty_example(self):
        raise NotImplementedError("TODO!")

    def _get_slice(self, t0_datetime_utc: datetime.datetime, center_osgb: Location) -> xr.Dataset:
        """Get a slice of sun position data.

        Note that, unlike most RawDataSources, SunPosition returns a Dataset, not a DataArray.
        This is necessary because SunPosition is made up of two data fields: azimuth & elevation.

        """
        # pvlib expects lat, lon, and a Pandas DatetimeIndex. Get lat lon:
        lat, lon = osgb_to_lat_lon(x=center_osgb.x, y=center_osgb.y)

        # Get DatetimeIndex:
        start_dt_ceil = self._get_start_dt_ceil(t0_datetime_utc)
        end_dt_ceil = self._get_end_dt_ceil(t0_datetime_utc)
        dt_index = pd.date_range(
            start=start_dt_ceil, end=end_dt_ceil, freq=self.sample_period_duration, name="time_utc"
        )

        # Compute solar azimuth and elevation:
        solpos = pvlib.solarposition.get_solarposition(
            time=dt_index,
            latitude=lat,
            longitude=lon,
            # Which `method` to use?
            # pyephem seemed to be a good mix between speed and ease but causes segfaults!
            # nrel_numba doesn't work when using multiple worker processes.
            # nrel_c is probably fastest but requires C code to be manually compiled:
            # https://midcdmz.nrel.gov/spa/
        )

        return solpos[["azimuth", "elevation"]].to_xarray()

    def _post_process(self, xr_data: xr.Dataset) -> xr.Dataset:
        # Normalise.
        xr_data["azimuth"] = (xr_data["azimuth"] - AZIMUTH_MEAN) / AZIMUTH_STD
        xr_data["elevation"] = (xr_data["elevation"] - ELEVATION_MEAN) / ELEVATION_STD
        return xr_data

    @staticmethod
    def to_numpy(xr_data: xr.Dataset) -> NumpyBatch:
        """Return a single example in a `NumpyBatch`."""
        example: NumpyBatch = {}
        example[BatchKey.solar_azimuth] = xr_data["azimuth"].values
        example[BatchKey.solar_elevation] = xr_data["elevation"].values
        return example
