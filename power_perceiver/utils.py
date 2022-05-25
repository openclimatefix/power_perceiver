from pathlib import Path
from typing import Union

import fsspec.asyn
import numpy as np
import pandas as pd
from pathy import Pathy


def datetime64_to_float(datetimes: np.ndarray, dtype=np.float64) -> np.ndarray:
    nums = datetimes.astype("datetime64[s]").astype(dtype)
    mask = np.isfinite(datetimes)
    return np.where(mask, nums, np.NaN)


def assert_num_dims(tensor, num_expected_dims: int):
    assert len(tensor.shape) == num_expected_dims, (
        f"Expected tensor to have {num_expected_dims} dims." f" Instead, shape={tensor.shape}"
    )


def is_sorted(array: np.ndarray) -> bool:
    """Return True if array is sorted in ascending order."""
    # Adapted from https://stackoverflow.com/a/47004507/732596
    if len(array) == 0:
        return False
    return np.all(array[:-1] <= array[1:])


def check_path_exists(path: Union[str, Path]):
    """Raise a FileNotFoundError if `path` does not exist.

    `path` can include wildcards.
    """
    if not path:
        raise FileNotFoundError("Not a valid path!")
    filesystem = get_filesystem(path)
    if not filesystem.exists(path):
        # Maybe `path` includes a wildcard? So let's use `glob` to check.
        # Try `exists` before `glob` because `glob` might be slower.
        files = filesystem.glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f"{path} does not exist!")


def get_filesystem(path: Union[str, Path]) -> fsspec.AbstractFileSystem:
    r"""Get the fsspect FileSystem from a path.

    For example, if `path` starts with `gs://` then return a gcsfs.GCSFileSystem.

    It is safe for `path` to include wildcards in the final filename.
    """
    path = Pathy(path)
    return fsspec.open(path.parent).fs


def sample_row_and_drop_row_from_df(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.Series, pd.DataFrame]:
    """Return sampled_row, dataframe_with_row_dropped."""
    assert not df.empty
    row_idx = rng.integers(low=0, high=len(df))
    row = df.iloc[row_idx]
    df = df.drop(row.name)
    return row, df


def set_fsspec_for_multiprocess() -> None:
    """
    Clear reference to the loop and thread.

    This is a nasty hack that was suggested but NOT recommended by the lead fsspec developer!

    This appears necessary otherwise gcsfs hangs when used after forking multiple worker processes.
    Only required for fsspec >= 0.9.0

    See:
    - https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    - https://github.com/fsspec/filesystem_spec/pull/963#issuecomment-1131709948

    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None
    fsspec.asyn._lock = None
