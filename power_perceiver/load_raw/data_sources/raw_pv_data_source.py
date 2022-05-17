import datetime
from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from power_perceiver.consts import Location
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    TimeseriesDataSource,
    ZarrDataSource,
)


@dataclass(kw_only=True)
class RawPVDataSource(
    # Surprisingly, Python's class hierarchy is defined right-to-left.
    # So the base class must go on the right.
    ZarrDataSource,
    TimeseriesDataSource,
    RawDataSource,
):
    """Load PV data directly from the intermediate PV Zarr store."""

    # For now, let's assume the PV data is always 5-minutely.
    # Later (WP3?), we'll want to experiment with lower temporal resolution satellite imagery.
    sample_period_duration: ClassVar[datetime.timedelta] = datetime.timedelta(minutes=5)

    def __post_init__(self):  # noqa: D105
        RawDataSource.__post_init__(self)
        TimeseriesDataSource.__post_init__(self)
        ZarrDataSource.__post_init__(self)
        # Load everything into RAM once (at init) rather than in each worker process.
        # This should be faster!
        self.load_everything_into_ram()

    def load_everything_into_ram(self) -> None:
        """Open AND load PV data into RAM."""
        raise NotImplementedError("TODO!")

    def get_osgb_location_for_example(self) -> Location:
        """Get a single random geographical location."""
        # Not planning to implement this just yet. To start with, let's just use `RawGSPDataSource`
        # to generate locations.
        raise NotImplementedError("TODO!")

    def get_empty_example(self) -> xr.Dataset:
        """Get an empty example.

        The returned Dataset does not include an `example` dimension.
        """
        raise NotImplementedError("TODO!")

    def _get_time_slice(
        self, xr_dataset: xr.Dataset, t0_datetime_utc: datetime.datetime
    ) -> xr.Dataset:
        raise NotImplementedError("TODO!")

    def _get_spatial_slice(self, xr_dataset: xr.Dataset, center_osgb: Location) -> xr.Dataset:
        raise NotImplementedError("TODO!")

    def _post_process(self, xr_dataset: xr.Dataset) -> xr.Dataset:
        # TODO: Normalise
        raise NotImplementedError("TODO!")

    @staticmethod
    def to_numpy(xr_data: xr.DataArray) -> NumpyBatch:
        """Return a single example, represented in a `NumpyBatch`."""
        raise NotImplementedError("TODO!")
