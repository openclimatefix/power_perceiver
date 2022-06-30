from dataclasses import dataclass


@dataclass
class DataSource:
    pre_processors: list
    post_processors: list


class ProductionPVDataSource(DataSource):
    pass


class SatelliteDataSource(DataSource):
    pass


class RemoveBadPVSystems:
    pass


class FillNighttimePVWithNaNs:
    pass


@dataclass
class NowcastingProductionDataset:
    data_sources: list


class FourierEncodeRelativeSpaceTime:
    pass


class SunPosition:
    pass


class PVNet:
    pass


@dataclass
class RunModel:
    model: PVNet


class DisableForecastsRunAtNight:
    pass


@dataclass
class ClipPredictedPowerToZeroAtNight:
    elevation_threshold_degrees: int


class SaveForecastsToDB:
    pass


def run_data_pipeline(pipeline: list, space_time_location_selector):
    pass


class Normalize:
    pass


class SaveBatchesToDisk:
    pass


class NormalizePV:
    pass


class EncodePVSystemsWithGSPID:
    pass


class Select15MinSatellite:
    pass


class NormalizeSatellite:
    pass


class PatchSatellite:
    pass


class EncodePixelsWithGSPID:
    pass


def now():
    pass


"""
Design objectives:

- Make the code as modular as possible.
- Make it super-easy to see all the data processing steps in one glance.
- Make it easy to configure different data processing for different models.
- Share code across training and production:
  - Data processors
  - DataSources that load from Zarr.

Definitely haven't fully thought this through yet!
"""


# -------------- Instantiate DataSources ------------
pv = ProductionPVDataSource(
    # Processing functions that run when a data source first opens the source data,
    # before any ML examples are selected:
    pre_processors=[
        RemoveBadPVSystems(),
        FillNighttimePVWithNaNs(),
        NormalizePV(),
        EncodePVSystemsWithGSPID(),
    ],
    # Post processors run after an example has been selected:
    post_processors=[SunPosition()],
)

satellite = SatelliteDataSource(
    pre_processors=[
        Select15MinSatellite(),
        NormalizeSatellite(),
        PatchSatellite(),
        EncodePixelsWithGSPID(),
    ],
    post_processors=[SunPosition()],
)

# -------------- Instantiate Dataset -----------------
input_dataset = NowcastingProductionDataset(
    data_sources=[pv, satellite],
)

# ------------ Build production data pipeline ----------
# (It maybe only works like this if we can predict national PV in one forward pass.)
# A `BatchML` object is passed from one step to the next.
production_data_pipeline = [
    input_dataset,
    # Processing functions which operate across multiple data sources:
    FourierEncodeRelativeSpaceTime(),
    # Saving batches to disk is entirely optional during production,
    # where it may be useful for debugging:
    # During training, SaveBatchesToDisk can be used to save pre-prepared batches.
    # And then `LoadBatchesFromDisk()` can be used to load batches into this pipeline.
    SaveBatchesToDisk(),
    # Run ML model:
    RunModel(model=PVNet()),
    # Post-process output from model:
    ClipPredictedPowerToZeroAtNight(elevation_threshold_degrees=5),
    DisableForecastsRunAtNight(),
    SaveForecastsToDB(),
]

run_data_pipeline(
    production_data_pipeline,
    space_time_location_selector=now,
)
