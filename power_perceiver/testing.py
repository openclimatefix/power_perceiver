"""Utilities to help with unit testing."""

import logging
import os
from pathlib import Path

from gcsfs import GCSFileSystem

from power_perceiver.consts import REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS, DataSourceName

_log = logging.getLogger(__name__)


def get_filename_of_batch_of_data_and_maybe_download(
    data_source_name: DataSourceName, batch_idx: int
) -> Path:
    """If the data is already downloaded locally then return the filename.

    The testing data is too large to include in the git repo so we publish
    a small amount of data to a public Google Cloud Storage bucket.

    If the data is not already downloaded then download and return the filename.

    Args:
        data_source_name (DataSourceName): e.g. DataSourceName.satellite
        batch_idx (int): 0 or 1.

    Returns:
        Path: The full path to the local data.
    """
    assert batch_idx in [0, 1], f"batch_idx must be 0 or 1, not {batch_idx}"
    local_data_path = _get_path_of_local_data_for_testing()
    local_path_for_data_source = local_data_path / data_source_name.value
    if not local_path_for_data_source.exists():
        _log.info(f"Creating path for local testing data: {local_path_for_data_source}")
        os.makedirs(local_path_for_data_source)
    nc_filename = f"{batch_idx:06d}.nc"
    local_batch_filename = local_path_for_data_source / nc_filename
    if not local_batch_filename.exists():
        remote_batch_filename = (
            REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS / data_source_name.value / nc_filename
        )
        _log.info(f"Downloading {remote_batch_filename}")
        fs = GCSFileSystem()
        fs.get(str(remote_batch_filename), str(local_batch_filename))
    return local_batch_filename


def _get_path_of_power_perceiver_package() -> Path:
    import power_perceiver

    return Path(power_perceiver.__file__).parents[1]


def _get_path_of_local_data_for_testing() -> Path:
    return _get_path_of_power_perceiver_package() / "data_for_testing"
