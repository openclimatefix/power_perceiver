"""Constants and Enums."""

from numbers import Number
from typing import NamedTuple

from pathy import Pathy

PV_TIME_AXIS = 1
PV_SYSTEM_AXIS = 2

Y_OSGB_MEAN = 357021.38
Y_OSGB_STD = 612920.2
X_OSGB_MEAN = 187459.94
X_OSGB_STD = 622805.44

SATELLITE_SPACER_LEN = 17  # Patch of 4x4 + 1 for surface height.
PV_SPACER_LEN = 18  # 16 for embedding dim + 1 for marker + 1 for history


class Location(NamedTuple):
    """Represent a spatial location."""

    x: Number
    y: Number


REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS = Pathy(
    "gs://ocf-public/data_for_unit_tests/prepared_ML_training_data"
)
