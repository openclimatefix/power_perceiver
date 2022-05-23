from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

from power_perceiver.consts import BatchKey
from power_perceiver.geospatial import osgb_to_lat_lon
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_prepared_batches.data_sources.sun import (
    AZIMUTH_MEAN,
    AZIMUTH_STD,
    ELEVATION_MEAN,
    ELEVATION_STD,
)


@dataclass
class SunPosition:
    """This is kind of a duplicate of the info in the Sun pre-prepared batch.

    But we don't have access to those pre-prepared batches when training directly
    from the Zarr! Hence we need this when training directly from Zarr!
    """

    def __call__(self, np_batch: NumpyBatch) -> NumpyBatch:
        """Sets `BatchKey.solar_azimuth_at_t0` and `BatchKey.solar_elevation_at_t0`."""
        y_osgb = np_batch[BatchKey.hrvsatellite_y_osgb]  # example, y, x
        x_osgb = np_batch[BatchKey.hrvsatellite_x_osgb]  # example, y, x
        time_utc = np_batch[BatchKey.hrvsatellite_time_utc]  # example, time

        # Get the time and position for the centre of the t0 frame:
        y_centre_idx = int(y_osgb.shape[1] // 2)
        x_centre_idx = int(y_osgb.shape[2] // 2)
        y_osgb_centre = y_osgb[:, y_centre_idx, x_centre_idx]  # Shape: (example,)
        x_osgb_centre = x_osgb[:, y_centre_idx, x_centre_idx]  # Shape: (example,)

        # Convert to the units that pvlib expects: lat, lon.
        lats, lons = osgb_to_lat_lon(x=x_osgb_centre, y=y_osgb_centre)

        # Loop round each example to get the Sun's elevation and azimuth:
        azimuth = np.full_like(time_utc, fill_value=np.NaN)
        elevation = np.full_like(time_utc, fill_value=np.NaN)
        for i, (lat, lon, dt) in enumerate(zip(lats, lons, time_utc)):
            dt = pd.to_datetime(dt, unit="s")
            dt = pd.DatetimeIndex([dt])  # pvlib expects a `pd.DatetimeIndex`.
            solpos = pvlib.solarposition.get_solarposition(
                time=dt,
                latitude=lat,
                longitude=lon,
                # Which `method` to use?
                # pyephem seemed to be a good mix between speed and ease but causes segfaults!
                # nrel_numba doesn't work when using multiple worker processes.
                # nrel_c is probably fastest but requires C code to be manually compiled:
                # https://midcdmz.nrel.gov/spa/
            )
            azimuth[i] = solpos["azimuth"]
            elevation[i] = solpos["elevation"]

        # Normalise.
        azimuth = (azimuth - AZIMUTH_MEAN) / AZIMUTH_STD
        elevation = (elevation - ELEVATION_MEAN) / ELEVATION_STD

        # Store.
        np_batch[BatchKey.solar_azimuth] = azimuth
        np_batch[BatchKey.solar_elevation] = elevation
        return np_batch
