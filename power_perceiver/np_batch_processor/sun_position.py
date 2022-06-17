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
    """Append the the Sun's azimuth and elevation.

    This is a duplicate of the info in the Sun pre-prepared batch.

    But we don't have access to those pre-prepared batches when training directly
    from the Zarr! Hence we need this when training directly from Zarr!
    """

    modality_name: str

    def __post_init__(self):
        assert self.modality_name in ["satellite", "gsp", "pv"]

    def __call__(self, np_batch: NumpyBatch) -> NumpyBatch:
        """Set `BatchKey.hrvsatellite_solar_azimuth` and `BatchKey.hrvsatellite_solar_elevation`.

        Or `BatchKey.gsp_solar_azimuth` or `BatchKey.gsp_solar_elevation`.
        """
        if self.modality_name == "satellite":
            y_osgb = np_batch[BatchKey.hrvsatellite_y_osgb]  # example, y, x
            x_osgb = np_batch[BatchKey.hrvsatellite_x_osgb]  # example, y, x
            time_utc = np_batch[BatchKey.hrvsatellite_time_utc]  # example, time

            # Get the time and position for the centre of the t0 frame:
            y_centre_idx = int(y_osgb.shape[1] // 2)
            x_centre_idx = int(y_osgb.shape[2] // 2)
            y_osgb_centre = y_osgb[:, y_centre_idx, x_centre_idx]  # Shape: (example,)
            x_osgb_centre = x_osgb[:, y_centre_idx, x_centre_idx]  # Shape: (example,)
        elif self.modality_name == "gsp":
            y_osgb_centre = np_batch[BatchKey.gsp_y_osgb]
            x_osgb_centre = np_batch[BatchKey.gsp_x_osgb]
            time_utc = np_batch[BatchKey.gsp_time_utc]
        elif self.modality_name == "pv":
            y_osgb_centre = np.nanmean(np_batch[BatchKey.pv_y_osgb], axis=1)
            x_osgb_centre = np.nanmean(np_batch[BatchKey.pv_x_osgb], axis=1)
            time_utc = np_batch[BatchKey.pv_time_utc]

        # Convert to the units that pvlib expects: lat, lon.
        lats, lons = osgb_to_lat_lon(x=x_osgb_centre, y=y_osgb_centre)

        # Loop round each example to get the Sun's elevation and azimuth:
        azimuth = np.full_like(time_utc, fill_value=np.NaN).astype(np.float32)
        elevation = np.full_like(time_utc, fill_value=np.NaN).astype(np.float32)
        for example_idx, (lat, lon) in enumerate(zip(lats, lons)):
            dt = pd.to_datetime(time_utc[example_idx], unit="s")
            dt = pd.DatetimeIndex(dt)  # pvlib expects a `pd.DatetimeIndex`.
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
            azimuth[example_idx] = solpos["azimuth"]
            elevation[example_idx] = solpos["elevation"]

        # Normalise.
        azimuth = (azimuth - AZIMUTH_MEAN) / AZIMUTH_STD
        elevation = (elevation - ELEVATION_MEAN) / ELEVATION_STD

        # Check
        assert np.isfinite(azimuth).all()
        assert np.isfinite(elevation).all()

        # Store.
        if self.modality_name == "satellite":
            np_batch[BatchKey.hrvsatellite_solar_azimuth] = azimuth
            np_batch[BatchKey.hrvsatellite_solar_elevation] = elevation
        elif self.modality_name == "gsp":
            np_batch[BatchKey.gsp_solar_azimuth] = azimuth
            np_batch[BatchKey.gsp_solar_elevation] = elevation
        elif self.modality_name == "pv":
            np_batch[BatchKey.pv_solar_azimuth] = azimuth
            np_batch[BatchKey.pv_solar_elevation] = elevation
        return np_batch
