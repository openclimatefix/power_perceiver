"""Constants and Enums."""

from enum import Enum

import numpy as np
import xarray as xr
from pathy import Pathy


class DataSourceName(Enum):  # noqa: D101
    gsp = "gsp"
    nwp = "nwp"
    opticalflow = "opticalflow"
    pv = "pv"
    satellite = "satellite"
    hrvsatellite = "hrvsatellite"
    sun = "sun"
    topographic = "topographic"


class BatchKey(Enum):
    """The names of the different elements of each batch.

    This is also where we document the exact shape of each element.

    This is basically a superset of DataSourceName, because each DataSource
    may be split into several different BatchKey elements. For example, the
    pv DataSource yields `pv` and `pv_system_row_number` BatchKeys.
    """

    # -------------- SATELLITE AND HRV ------------------------------
    satellite = "satellite"
    hrvsatellite = "hrvsatellite"
    hrvsatellite_angle_in_degrees_from_pv_system = "hrvsatellite_angle_in_degrees_from_pv_system"
    hrvsatellite_distance_in_meters_from_pv_system = (
        "hrvsatellite_distance_in_meters_from_pv_system"
    )
    # -------------- PV ---------------------------------------------
    pv = "pv"
    pv_system_row_number = "pv_system_row_number"


XarrayBatch = dict[DataSourceName, xr.Dataset]
NumpyBatch = dict[BatchKey, np.ndarray]


REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS = Pathy(
    "gs://ocf-public/data_for_unit_tests/prepared_ML_training_data"
)
