import datetime
import logging
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import xarray as xr

from power_perceiver.consts import Location
from power_perceiver.geospatial import lat_lon_to_osgb
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    TimeseriesDataSource,
)

_log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RawPVDataSource(
    # Surprisingly, Python's class hierarchy is defined right-to-left.
    # So the base class must go on the right.
    TimeseriesDataSource,
    RawDataSource,
):
    """Load PV data directly from the intermediate PV Zarr store.

    Attributes:
        _pv_power: pd.DataFrame
        _pv_metadata: pd.DataFrame
        _pv_capacity: pd.Series (index is the PV system ID. Values are the capacity.)
    """

    pv_power_filename: str
    pv_metadata_filename: str

    # For now, let's assume the PV data is always 5-minutely.
    # Later (WP3?), we'll want to experiment with lower temporal resolution satellite imagery.
    sample_period_duration: ClassVar[datetime.timedelta] = datetime.timedelta(minutes=5)

    def __post_init__(self):  # noqa: D105
        RawDataSource.__post_init__(self)
        TimeseriesDataSource.__post_init__(self)
        # Load everything into RAM once (at init) rather than in each worker process.
        # This should be faster!
        self.load_everything_into_ram()

    def load_everything_into_ram(self) -> None:
        """Open AND load PV data into RAM."""
        self._pv_power = _load_pv_power(self.pv_power_filename)
        self._pv_metadata = _load_metadata(self.pv_metadata_filename)
        self._pv_metadata, self._pv_power = _align_pv_system_ids(self._pv_metadata, self._pv_power)

    def get_osgb_location_for_example(self) -> Location:
        """Get a single random geographical location."""
        raise NotImplementedError(
            "Not planning to implement this just yet. To start with, let's use"
            " `RawSatelliteDataSource` and/or `RawGSPDataSource` to generate locations."
        )

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
        """Return a single example in a `NumpyBatch`."""
        raise NotImplementedError("TODO!")


def _load_pv_metadata(filename: str) -> pd.DataFrame:
    """Return pd.DataFrame of PV metadata.

    Shape of the returned pd.DataFrame for Passiv PV data:
        Index: ss_id (Sheffield Solar ID)
        Columns: llsoacd, orientation, tilt, kwp, operational_at,
            latitude, longitude, system_id, x_osgb, y_osgb
    """
    _log.info(f"Loading PV metadata from {filename}")
    pv_metadata = pd.read_csv(filename, index_col="ss_id").drop(columns="Unnamed: 0")
    _log.info(f"Found {len(pv_metadata)} PV systems in {filename}")

    # drop any systems with no lon or lat:
    pv_metadata.dropna(subset=["longitude", "latitude"], how="any", inplace=True)

    _log.debug(f"Found {len(pv_metadata)} PV systems with locations")

    pv_metadata["x_osgb"], pv_metadata["y_osgb"] = lat_lon_to_osgb(
        latitude=pv_metadata["latitude"], longitude=pv_metadata["longitude"]
    )

    # Remove PV systems outside the geospatial boundary of the satellite data:
    GEO_BOUNDARY_OSGB = {
        "WEST": -238_000,
        "EAST": 856_000,
        "NORTH": 1_222_000,
        "SOUTH": -184_000,
    }

    pv_metadata = pv_metadata[
        (pv_metadata.x_osgb >= GEO_BOUNDARY_OSGB["WEST"])
        & (pv_metadata.x_osgb <= GEO_BOUNDARY_OSGB["EAST"])
        & (pv_metadata.y_osgb <= GEO_BOUNDARY_OSGB["NORTH"])
        & (pv_metadata.y_osgb >= GEO_BOUNDARY_OSGB["SOUTH"])
    ]

    _log.info(f"Found {len(pv_metadata)} PV systems after filtering.")
    return pv_metadata
