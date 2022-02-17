from power_perceiver.consts import DataSourceName
from power_perceiver.data_loader.data_loader import DataLoader
from power_perceiver.data_loader.satellite import SatelliteDataLoader

DATA_SOURCE_NAME_TO_LOADER_CLASS: dict[DataSourceName, DataLoader] = {
    DataSourceName.satellite: SatelliteDataLoader,
}
