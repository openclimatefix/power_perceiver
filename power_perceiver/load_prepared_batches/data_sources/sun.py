import numpy as np
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import (
    NumpyBatch,
    PreparedDataSource,
)

ELEVATION_MEAN = 37.4
ELEVATION_STD = 12.7
AZIMUTH_MEAN = 177.7
AZIMUTH_STD = 41.7


class Sun(PreparedDataSource):
    def process_before_transforms(self, dataset: xr.Dataset) -> xr.Dataset:
        # None of this will be necessary once this is implemented:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/635

        # Drop redundant coordinates (these are redundant because they
        # just repeat the contents of each *dimension*):
        dataset = dataset.drop_vars(["example", "time_index"])

        # Rename coords to be more explicit about exactly what some coordinates hold:
        dataset = dataset.rename_vars({"time": "time_utc"})

        # Rename dimensions. Standardize on the singular (time, channel, etc.).
        # Remove redundant "index" from the dim name. These are *dimensions* so,
        # by definition, they are indicies!
        dataset = dataset.rename_dims({"time_index": "time"})

        dataset = dataset.set_coords("time_utc")
        return dataset

    @staticmethod
    def to_numpy(dataset: xr.Dataset) -> NumpyBatch:
        """This is called from Dataset.__getitem__.

        Sets `BatchKey.solar_azimuth` and `BatchKey.solar_elevation`.

        This processes this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        batch: NumpyBatch = {}

        solar_azimuth = dataset["azimuth"].astype(np.float32).values
        solar_azimuth -= AZIMUTH_MEAN
        solar_azimuth /= AZIMUTH_STD
        batch[BatchKey.hrvsatellite_solar_azimuth] = solar_azimuth

        solar_elevation = dataset["elevation"].astype(np.float32).values
        solar_elevation -= ELEVATION_MEAN
        solar_elevation /= ELEVATION_STD
        batch[BatchKey.hrvsatellite_solar_elevation] = solar_elevation

        return batch
