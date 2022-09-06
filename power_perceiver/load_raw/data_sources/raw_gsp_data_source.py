import datetime
import logging
from dataclasses import dataclass
from typing import ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.consts import Location
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    TimeseriesDataSource,
)
from power_perceiver.utils import check_path_exists, datetime64_to_float, select_time_periods

_log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RawGSPDataSource(
    # Surprisingly, Python's class hierarchy is defined right-to-left.
    # So the base class must go on the right.
    TimeseriesDataSource,
    RawDataSource,
):
    """
    Data source for GSP (Grid Supply Point) PV Data.

    Load GSP data directly from the intermediate GSP Zarr store.

    30 mins data is taken from 'PV Live' from https://www.solar.sheffield.ac.uk/pvlive/
    meta data is taken from ESO.  PV Live estimates the total PV power generation for each
    Grid Supply Point region.
    """

    gsp_pv_power_zarr_path: str
    gsp_id_to_region_id_filename: str
    sheffield_solar_region_path: str
    threshold_mw: int = 0

    sample_period_duration: ClassVar[datetime.timedelta] = datetime.timedelta(minutes=30)

    def __post_init__(self):  # noqa: D105
        self._sanity_check_args()
        RawDataSource.__post_init__(self)
        TimeseriesDataSource.__post_init__(self)
        # Load everything into RAM once (at init) rather than in each worker process.
        # This should be faster than loading from disk in every worker!
        self.load_everything_into_ram()
        self.empty_example = self._get_empty_example()

    def _sanity_check_args(self):  # noqa: D105
        check_path_exists(self.gsp_pv_power_zarr_path)
        check_path_exists(self.gsp_id_to_region_id_filename)
        check_path_exists(self.sheffield_solar_region_path)
        assert self.threshold_mw >= 0

    def load_everything_into_ram(self) -> None:
        """Open AND load GSP data into RAM."""
        gsp_id_to_shape = _get_gsp_id_to_shape(
            self.gsp_id_to_region_id_filename, self.sheffield_solar_region_path
        )
        self._gsp_id_to_shape = gsp_id_to_shape  # Save, mostly for plotting to check all is fine!

        # Load GSP generation xr.Dataset:
        gsp_pv_power_mw_ds = xr.open_dataset(self.gsp_pv_power_zarr_path, engine="zarr")
        gsp_pv_power_mw_ds = select_time_periods(
            xr_data=gsp_pv_power_mw_ds,
            time_periods=self.time_periods,
            dim_name="datetime_gmt",
        )
        gsp_pv_power_mw_ds = gsp_pv_power_mw_ds.load()

        # Ensure the centroids have the same GSP ID index as the GSP PV power:
        gsp_id_to_shape = gsp_id_to_shape.loc[gsp_pv_power_mw_ds.gsp_id]

        data_array = _put_gsp_data_into_an_xr_dataarray(
            gsp_pv_power_mw=gsp_pv_power_mw_ds.generation_mw.data.astype(np.float32),
            time_utc=gsp_pv_power_mw_ds.datetime_gmt.data,
            gsp_id=gsp_pv_power_mw_ds.gsp_id.data,
            # TODO: Try using `gsp_id_to_shape.geometry.envelope.centroid`. See issue #76.
            x_osgb=gsp_id_to_shape.geometry.centroid.x.astype(np.float32),
            y_osgb=gsp_id_to_shape.geometry.centroid.y.astype(np.float32),
            capacity_mwp=gsp_pv_power_mw_ds.installedcapacity_mwp.data.astype(np.float32),
            t0_idx=self.t0_idx,
        )

        del gsp_id_to_shape, gsp_pv_power_mw_ds

        # Select GSPs with sufficient installed PV capacity.
        # We take the `max` across time, because we don't want to drop GSPs which
        # do have installed PV capacity now, but didn't have PV power a while ago.
        # Although, in practice, swapping between `min` and `max` doesn't actually
        # seem to make any difference to the number of GSPs dropped :)
        gsp_id_mask = data_array.capacity_mwp.max(dim="time_utc") > self.threshold_mw
        data_array = data_array.isel(gsp_id=gsp_id_mask)
        data_array = data_array.dropna(dim="time_utc", how="all")

        _log.debug(
            f"There are {len(data_array.gsp_id)} GSPs left after we dropped"
            f" {(~gsp_id_mask).sum().data} GSPs because they had less than"
            f" {self.threshold_mw} MW of installed PV capacity."
            f" Total number of timesteps: {len(data_array.time_utc)}"
        )

        self._data_in_ram = data_array

    @property
    def num_gsps(self) -> int:
        return len(self.data_in_ram.gsp_id)

    def get_osgb_location_for_example(self) -> Location:
        """Get a single random geographical location."""
        random_gsp_idx = self.rng.integers(low=0, high=len(self.data_in_ram.gsp_id))
        return self.get_osgb_location_for_gsp_idx(random_gsp_idx)

    def get_osgb_location_for_gsp_idx(self, gsp_idx: int) -> Location:
        random_gsp = self.data_in_ram.isel(gsp_id=gsp_idx)
        return Location(x=random_gsp.x_osgb.item(), y=random_gsp.y_osgb.item())

    def _get_empty_example(self) -> xr.DataArray:
        """Get an empty example.

        The returned DataArray does not include an `example` dimension.
        """
        gsp_pv_power_mw = np.full(
            shape=(self.total_seq_length, 1), fill_value=np.NaN, dtype=np.float32
        )
        time_utc = np.full(shape=self.total_seq_length, fill_value=np.NaN, dtype=np.float32)
        time_utc = pd.DatetimeIndex(time_utc)
        gsp_meta = np.full(shape=1, fill_value=np.NaN, dtype=np.float32)

        return _put_gsp_data_into_an_xr_dataarray(
            gsp_pv_power_mw=gsp_pv_power_mw,
            time_utc=time_utc,
            gsp_id=gsp_meta,
            x_osgb=gsp_meta,
            y_osgb=gsp_meta,
            capacity_mwp=gsp_pv_power_mw,
            t0_idx=self.t0_idx,
        )

    def _get_spatial_slice(self, xr_data: xr.DataArray, center_osgb: Location) -> xr.DataArray:
        """Just return data for 1 GSP: The GSP whose centroid is (almost) equal to center_osgb."""
        mask = np.isclose(xr_data.x_osgb, center_osgb.x) & np.isclose(xr_data.y_osgb, center_osgb.y)
        return xr_data.isel(gsp_id=mask)

    def _post_process(self, xr_data: xr.DataArray) -> xr.DataArray:
        xr_data = xr_data / xr_data.capacity_mwp
        return xr_data

    @staticmethod
    def to_numpy(xr_data: xr.DataArray) -> NumpyBatch:
        """Return a single example in a `NumpyBatch`."""
        example: NumpyBatch = {}

        example[BatchKey.gsp] = xr_data.values
        example[BatchKey.gsp_t0_idx] = xr_data.attrs["t0_idx"]
        example[BatchKey.gsp_id] = xr_data.gsp_id.values
        example[BatchKey.gsp_capacity_mwp] = xr_data.isel(time_utc=0)["capacity_mwp"].values

        # Coordinates
        example[BatchKey.gsp_time_utc] = datetime64_to_float(xr_data["time_utc"].values)
        for batch_key, dataset_key in (
            (BatchKey.gsp_y_osgb, "y_osgb"),
            (BatchKey.gsp_x_osgb, "x_osgb"),
        ):
            values = xr_data[dataset_key].values
            # Expand dims so EncodeSpaceTime works!
            example[batch_key] = values  # np.expand_dims(values, axis=1)

        return example


def _get_gsp_id_to_shape(
    gsp_id_to_region_id_filename: str, sheffield_solar_region_path: str
) -> gpd.GeoDataFrame:
    # Load mapping from GSP ID to Sheffield Solar region ID:
    gsp_id_to_region_id = pd.read_csv(
        gsp_id_to_region_id_filename,
        usecols=["gsp_id", "region_id"],
        dtype={"gsp_id": np.int64, "region_id": np.int64},
    )

    # Load Sheffield Solar region shapes (which are already in OSGB36 CRS).
    ss_regions = gpd.read_file(sheffield_solar_region_path)

    # Merge, so we have a mapping from GSP ID to SS region shape:
    gsp_id_to_shape = (
        ss_regions.merge(gsp_id_to_region_id, left_on="RegionID", right_on="region_id")
        .set_index("gsp_id")[["geometry"]]
        .sort_index()
    )

    # Some GSPs are represented by multiple shapes. To find the correct centroid,
    # we need to find the spatial union of those regions, and then find the centroid
    # of those spatial unions. `dissolve(by="gsp_id")` groups by "gsp_id" and gets
    # the spatial union.
    return gsp_id_to_shape.dissolve(by="gsp_id")


def _put_gsp_data_into_an_xr_dataarray(
    gsp_pv_power_mw: np.ndarray,
    time_utc: np.ndarray,
    gsp_id: np.ndarray,
    x_osgb: np.ndarray,
    y_osgb: np.ndarray,
    capacity_mwp: np.ndarray,
    t0_idx: int,
) -> xr.DataArray:
    # Convert to xr.DataArray:
    data_array = xr.DataArray(
        gsp_pv_power_mw,
        coords=(("time_utc", time_utc), ("gsp_id", gsp_id)),
        name="gsp_pv_power_mw",
    )
    data_array = data_array.assign_coords(
        x_osgb=("gsp_id", x_osgb),
        y_osgb=("gsp_id", y_osgb),
        capacity_mwp=(("time_utc", "gsp_id"), capacity_mwp),
    )
    data_array.attrs["t0_idx"] = t0_idx
    return data_array
