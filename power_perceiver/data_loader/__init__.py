from power_perceiver.consts import DataSourceName
from power_perceiver.data_loader.data_loader import DataLoader
from power_perceiver.data_loader.pv import PVLoader
from power_perceiver.data_loader.satellite import HRVSatelliteLoader

# For now, every DataSourceName is mapped to DataLoader. In the future,
# each DataSourceName could be mapped to a subclass of DataLoader.
DATA_SOURCE_NAME_TO_LOADER_CLASS: dict[DataSourceName, DataLoader] = {
    data_source_name: DataLoader for data_source_name in DataSourceName
}
DATA_SOURCE_NAME_TO_LOADER_CLASS[DataSourceName.hrvsatellite] = HRVSatelliteLoader
DATA_SOURCE_NAME_TO_LOADER_CLASS[DataSourceName.pv] = PVLoader
