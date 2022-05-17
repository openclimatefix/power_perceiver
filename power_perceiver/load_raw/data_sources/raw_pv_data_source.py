import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, Union

import fsspec
import numpy as np
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
        _data_in_ram: xr.DataArray
            The data is the 5-minutely PV power in Watts.
            Dimension coordinates: time_utc, pv_system_id
            Additional coordinates: x_osgb, y_osgb, capacity_wp
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
        pv_power_watts, pv_capacity_wp = _load_pv_power_watts_and_capacity_wp(
            self.pv_power_filename, start_date=self.start_date, end_date=self.end_date
        )
        pv_metadata = _load_pv_metadata(self.pv_metadata_filename)
        pv_metadata, pv_power_watts = _align_pv_system_ids(pv_metadata, pv_power_watts)
        pv_capacity_wp = pv_capacity_wp.loc[pv_power_watts.columns]

        # Convert to an xarray DataArray, which gets saved to `self._data_in_ram`
        data_array = xr.DataArray(
            data=pv_power_watts.values,
            coords=(("time_utc", pv_power_watts.index), ("pv_system_id", pv_power_watts.columns)),
            name="pv_power_watts",
        )
        data_array = data_array.assign_coords(
            x_osgb=("pv_system_id", pv_metadata.x_osgb.astype(np.float32)),
            y_osgb=("pv_system_id", pv_metadata.y_osgb.astype(np.float32)),
            capacity_wp=("pv_system_id", pv_capacity_wp),
        )
        self._data_in_ram = data_array

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

    def _get_spatial_slice(self, xr_dataset: xr.Dataset, center_osgb: Location) -> xr.Dataset:
        raise NotImplementedError("TODO!")

    def _post_process(self, xr_dataset: xr.Dataset) -> xr.Dataset:
        # TODO: Normalise
        raise NotImplementedError("TODO!")

    @staticmethod
    def to_numpy(xr_data: xr.DataArray) -> NumpyBatch:
        """Return a single example in a `NumpyBatch`."""
        raise NotImplementedError("TODO!")


# Adapted from nowcasting_dataset.data_sources.pv.pv_data_source
def _load_pv_power_watts_and_capacity_wp(
    filename: Union[str, Path],
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return pv_power_watts, pv_capacity_wp."""
    _log.info(f"Loading solar PV power data from {filename} from {start_date=} to {end_date=}.")

    # Load data in a way that will work in the cloud and locally:
    with fsspec.open(filename, mode="rb") as file:
        pv_power_ds = xr.open_dataset(file, engine="h5netcdf")
        pv_capacity_wp = pv_power_ds.max()
        pv_capacity_wp = pv_capacity_wp.to_pandas().astype(np.float32)
        pv_power_ds = pv_power_ds.sel(datetime=slice(start_date, end_date))
        pv_power_watts = pv_power_ds.to_dataframe().astype(np.float32)

    pv_capacity_wp.index = [np.int32(col) for col in pv_capacity_wp.index]

    _log.info(
        "Before filtering:"
        f" Found {len(pv_power_watts)} PV power datetimes."
        f" Found {len(pv_power_watts.columns)} PV power PV system IDs."
    )

    # Drop columns and rows with all NaNs.
    pv_power_watts.dropna(axis="columns", how="all", inplace=True)
    pv_power_watts.dropna(axis="index", how="all", inplace=True)
    pv_power_watts = pv_power_watts.clip(lower=0, upper=5e7)
    # Convert the pv_system_id column names from strings to ints:
    pv_power_watts.columns = [np.int32(col) for col in pv_power_watts.columns]

    if "passiv" not in filename:
        _log.warning("Converting timezone. ARE YOU SURE THAT'S WHAT YOU WANT TO DO?")
        pv_power_watts = (
            pv_power_watts.tz_localize("Europe/London").tz_convert("UTC").tz_convert(None)
        )

    pv_power_watts = _drop_pv_systems_which_produce_overnight(pv_power_watts)

    # Resample to 5-minutely and interpolate up to 15 minutes ahead.
    # TODO: Issue #301: Give users the option to NOT resample (because Perceiver IO
    # doesn't need all the data to be perfectly aligned).
    pv_power_watts = pv_power_watts.resample("5T").interpolate(method="time", limit=3)
    pv_power_watts.dropna(axis="index", how="all", inplace=True)
    _log.info(f"pv_power = {pv_power_watts.values.nbytes / 1e6:,.1f} MBytes.")
    _log.info(f"After resampling to 5 mins, there are now {len(pv_power_watts)} pv power datetimes")

    _log.info(
        "After filtering:"
        f" Found {len(pv_power_watts)} PV power datetimes."
        f" Found {len(pv_power_watts.columns)} PV power PV system IDs."
    )

    # Sanity checks:
    assert not pv_power_watts.columns.duplicated().any()
    assert not pv_power_watts.index.duplicated().any()
    return pv_power_watts, pv_capacity_wp


# Adapted from nowcasting_dataset.data_sources.pv.pv_data_source
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


# From nowcasting_dataset.data_sources.pv.pv_data_source
def _drop_pv_systems_which_produce_overnight(pv_power_watts: pd.DataFrame) -> pd.DataFrame:
    """Drop systems which produce power over night.

    Args:
        pv_power_watts: Un-normalised.
    """
    # TODO: Of these bad systems, 24647, 42656, 42807, 43081, 51247, 59919
    # might have some salvagable data?
    NIGHT_YIELD_THRESHOLD = 0.4
    night_hours = [22, 23, 0, 1, 2]
    pv_power_normalised = pv_power_watts / pv_power_watts.max()
    night_mask = pv_power_normalised.index.hour.isin(night_hours)
    pv_power_at_night_normalised = pv_power_normalised.loc[night_mask]
    pv_above_threshold_at_night = (pv_power_at_night_normalised > NIGHT_YIELD_THRESHOLD).any()
    bad_systems = pv_power_normalised.columns[pv_above_threshold_at_night]
    _log.info(f"{len(bad_systems)} bad PV systems found and removed!")
    return pv_power_watts.drop(columns=bad_systems)


# From nowcasting_dataset.data_sources.pv.pv_data_source
def _align_pv_system_ids(
    pv_metadata: pd.DataFrame, pv_power: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only pick PV systems for which we have metadata."""
    pv_system_ids = pv_metadata.index.intersection(pv_power.columns)
    pv_system_ids = np.sort(pv_system_ids)
    pv_power = pv_power[pv_system_ids]
    pv_metadata = pv_metadata.loc[pv_system_ids]
    return pv_metadata, pv_power
