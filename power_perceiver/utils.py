import numpy as np


def datetime64_to_int(datetimes: np.ndarray) -> np.ndarray:
    return datetimes.astype("datetime64[s]").astype(np.int32)
