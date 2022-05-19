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
from power_perceiver.utils import check_path_exists

_log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RawPVDataSource(
    # Surprisingly, Python's class hierarchy is defined right-to-left.
    # So the base class must go on the right.
    TimeseriesDataSource,
    RawDataSource,
):
    """Load PV data directly from the intermediate PV Zarr store.

    Args:
        pv_power_filename:
        pv_metadata_filename:
        roi_height_meters: The height of the region of interest (ROI) when creating examples.
            For PV, we use meters (not pixels) because PV isn't an image.
            Must be at least 1,000 meters.
        roi_width_meters:
        n_pv_systems_per_example: Each example will have exactly this number of PV systems.
            Randomly select PV systems for each example. If there are less PV systems available
            than requested, then randomly sample with duplicates allowed, whilst ensuring all
            available PV systems are used.

    Attributes:
        empty_example: xr.DataArray: An example of the correct shape, but where data and coords
            are all NaNs!
        _data_in_ram: xr.DataArray
            The data is the 5-minutely PV power in Watts.
            Dimension coordinates: time_utc, pv_system_id
            Additional coordinates: x_osgb, y_osgb, capacity_wp
    """

    pv_power_filename: str
    pv_metadata_filename: str
    roi_height_meters: int
    roi_width_meters: int
    n_pv_systems_per_example: int

    # For now, let's assume the PV data is always 5-minutely, even though some PVOutput.org
    # PV systems report data at 15-minutely intervals. For now, let's just interpolate
    # 15-minutely data to 5-minutely. Later (WP3?) we could experiment with giving the model
    # the "raw" (un-interpolated) 15-minutely PV data.
    sample_period_duration: ClassVar[datetime.timedelta] = datetime.timedelta(minutes=5)

    def __post_init__(self):  # noqa: D105
        self._sanity_check_args()
        RawDataSource.__post_init__(self)
        TimeseriesDataSource.__post_init__(self)
        # Load everything into RAM once (at init) rather than in each worker process.
        # This should be faster than loading from disk in every worker!
        self.load_everything_into_ram()
        self.empty_example = self._get_empty_example()

    def _sanity_check_args(self) -> None:
        check_path_exists(self.pv_power_filename)
        check_path_exists(self.pv_metadata_filename)
        assert self.roi_height_meters > 1_000
        assert self.roi_width_meters > 1_000
        assert self.n_pv_systems_per_example > 0

    def load_everything_into_ram(self) -> None:
        """Open AND load PV data into RAM."""
        # Load pd.DataFrame of power and pd.Series of capacities:
        pv_power_watts, pv_capacity_wp = _load_pv_power_watts_and_capacity_wp(
            self.pv_power_filename, start_date=self.start_date, end_date=self.end_date
        )
        pv_metadata = _load_pv_metadata(self.pv_metadata_filename)
        # Ensure pv_metadata, pv_power_watts, and pv_capacity_wp all have the same set of
        # PV system IDs, in the same order:
        pv_metadata, pv_power_watts = _intersection_of_pv_system_ids(pv_metadata, pv_power_watts)
        pv_capacity_wp = pv_capacity_wp.loc[pv_power_watts.columns]

        self._data_in_ram = _put_pv_data_into_an_xr_dataarray(
            pv_power_watts=pv_power_watts,
            y_osgb=pv_metadata.y_osgb.astype(np.float32),
            x_osgb=pv_metadata.x_osgb.astype(np.float32),
            capacity_wp=pv_capacity_wp.values,
        )

    def get_osgb_location_for_example(self) -> Location:
        """Get a single random geographical location."""
        raise NotImplementedError(
            "Not planning to implement this just yet. To start with, let's use"
            " `RawSatelliteDataSource` and/or `RawGSPDataSource` to generate locations."
        )

    def _get_spatial_slice(self, xr_data: xr.DataArray, center_osgb: Location) -> xr.DataArray:

        half_roi_width_meters = self.roi_width_meters // 2
        half_roi_height_meters = self.roi_height_meters // 2

        left = center_osgb.x - half_roi_width_meters
        right = center_osgb.x + half_roi_width_meters
        top = center_osgb.y + half_roi_height_meters
        bottom = center_osgb.y - half_roi_height_meters

        # Sanity check!
        min_x_osgb = xr_data.x_osgb.min()
        max_x_osgb = xr_data.x_osgb.max()
        min_y_osgb = xr_data.y_osgb.min()
        max_y_osgb = xr_data.y_osgb.max()
        assert left >= min_x_osgb, f"{left=} must be >= {min_x_osgb=}"
        assert right <= max_x_osgb, f"{right=} must be <= {max_x_osgb=}"
        assert top <= max_y_osgb, f"{top=} must be <= {max_y_osgb=}"
        assert bottom >= min_y_osgb, f"{bottom=} must be >= {min_y_osgb=}"

        # Select data in the region of interest:
        pv_system_id_mask = (
            (left <= xr_data.x_osgb)
            & (xr_data.x_osgb <= right)
            & (xr_data.y_osgb <= top)
            & (bottom <= xr_data.y_osgb)
        )

        selected_data = xr_data.isel(pv_system_id=pv_system_id_mask)
        return self._ensure_n_pv_systems_per_example(selected_data)

    def _ensure_n_pv_systems_per_example(self, selected_data: xr.DataArray) -> xr.DataArray:
        """Ensure there are always `self.n_pv_systems_per_example` PV systems."""
        if len(selected_data.pv_system_id) > self.n_pv_systems_per_example:
            # More PV systems are available than we need. Reduce by randomly sampling:
            subset_of_pv_system_ids = self.rng.choice(
                selected_data.pv_system_id,
                size=self.n_pv_systems_per_example,
                replace=False,
            )
            selected_data = selected_data.sel(pv_system_id=subset_of_pv_system_ids)
        elif len(selected_data.pv_system_id) < self.n_pv_systems_per_example:
            # If we just used `choice(replace=True)` then there's a high chance
            # that the output won't include every available PV system but instead
            # will repeat some PV systems at the expense of leaving some on the table.
            # TODO: Don't repeat PV systems. Instead, pad with NaNs and mask the loss. Issue #73.
            n_random_pv_systems = self.n_pv_systems_per_example - len(selected_data.pv_system_id)
            allow_replacement = n_random_pv_systems > len(selected_data.pv_system_id)
            random_pv_system_ids = self.rng.choice(
                selected_data.pv_system_id,
                size=n_random_pv_systems,
                replace=allow_replacement,
            )
            selected_data = xr.concat(
                (selected_data, selected_data.sel(pv_system_id=random_pv_system_ids)),
                dim="pv_system_id",
            )
        return selected_data

    def _post_process(self, xr_data: xr.DataArray) -> xr.DataArray:
        xr_data = xr_data / xr_data.capacity_wp
        assert np.isfinite(xr_data).all()
        return xr_data

    def _get_empty_example(self) -> xr.DataArray:
        """Return a single example of the correct shape but where data & coords are all NaN."""
        pass  # TODO!

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
    """Return pv_power_watts, pv_capacity_wp.

    The capacities are the max across the *entire* dataset, and so is independent of the
    `start_date` and `end_date`. This is important so we always normalise PV data
    in the same way for training and testing sets!
    """

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
def _intersection_of_pv_system_ids(
    pv_metadata: pd.DataFrame, pv_power: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only pick PV systems for which we have metadata."""
    pv_system_ids = pv_metadata.index.intersection(pv_power.columns)
    pv_system_ids = np.sort(pv_system_ids)
    pv_power = pv_power[pv_system_ids]
    pv_metadata = pv_metadata.loc[pv_system_ids]
    return pv_metadata, pv_power


def _put_pv_data_into_an_xr_dataarray(
    pv_power_watts: pd.DataFrame,
    y_osgb: pd.Series,
    x_osgb: pd.Series,
    capacity_wp: pd.Series,
) -> xr.DataArray:
    """Convert to an xarray DataArray."""
    data_array = xr.DataArray(
        data=pv_power_watts.values,
        coords=(("time_utc", pv_power_watts.index), ("pv_system_id", pv_power_watts.columns)),
        name="pv_power_watts",
    )
    data_array = data_array.assign_coords(
        x_osgb=("pv_system_id", x_osgb),
        y_osgb=("pv_system_id", y_osgb),
        capacity_wp=("pv_system_id", capacity_wp),
    )
    return data_array
