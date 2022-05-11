from dataclasses import dataclass, field
from numbers import Number

import numpy as np

from power_perceiver.consts import BatchKey
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch


@dataclass
class EncodeSpaceTime:
    """Encode space and time coords in a way that Perceiver understands :)

    The broad approach is:

    For each coordinate in (x_osgb, y_osgb, time_utc):

    1. Compute the min and max across all modalities.
    2. For each modality, rescale the coordinate to [0, 1] using the min and max across all
       modalities.
    3. Compute Fourier features for each modality.

    The end result is that the NumpyBatch will now include:

    - <modality name>_<coordinate_name>_fourier

    init args:
        lengths: The approximate lengths of each dimension across an example. For example,
            if the dimension is x_osgb then the length will be the approximate distance in meters
            from the left to the right side of an average example. If the dimension is time_utc
            then the length is the approximate total duration in seconds of the example.
            The keys must be x_osgb, y_osgb, or time_utc.
            We need to use constant lengths across all examples so the spatial encoding
            is always proportional to the "real world" length across all examples.
            If we didn't do that, a spatial encoding of 1 would represent different "real world"
            distances across examples. This would almost certainly be harmful, especially
            because we're expecting the model to learn to do some basic geometry!
        n_fourier_features_per_dim:
    """

    lengths: dict[str, Number] = field(
        # Using lambda with default factory when using a mutable default value adapted from:
        # https://stackoverflow.com/a/52064202/732596
        default_factory=lambda: dict(x_osgb=120_000, y_osgb=200_000, time_utc=60 * 5 * 31)
    )
    n_fourier_features_per_dim: int = 8

    def __call__(self, np_batch: NumpyBatch) -> NumpyBatch:
        return get_spatial_and_temporal_fourier_features(
            np_batch=np_batch,
            lengths=self.lengths,
            n_fourier_features_per_dim=self.n_fourier_features_per_dim,
        )


def get_spatial_and_temporal_fourier_features(
    np_batch: NumpyBatch,
    lengths: dict[str, Number],
    n_fourier_features_per_dim: int = 8,
) -> NumpyBatch:
    """Add fourier features for x_osgb, y_osgb and time_utc."""

    rescaled_coords: dict[str, np.ndarray] = _rescale_coords_for_all_dims_to_approx_0_to_1(
        np_batch=np_batch, lengths=lengths
    )

    for key, coords in rescaled_coords.items():
        new_key = key.replace("rescaled", "fourier")
        new_key = BatchKey[new_key]
        np_batch[new_key] = compute_fourier_features(
            coords, n_fourier_features=n_fourier_features_per_dim
        )

    return np_batch


def compute_fourier_features(
    array: np.ndarray, n_fourier_features: int = 8, min_freq: float = 2, max_freq: float = 8
) -> np.ndarray:
    """Compute Fourier features for a single dimension, across all examples in a batch.

    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        array: np.ndarray with values roughly in the range [0, 1].
            The values don't have to be *exactly* in the range [0, 1] because sine and cosine
            handle values below 0 and above 2*pi.
            For the time dimension, the shape will be (batch_size, n_timesteps).
            For spatial dimensions, the shape might be (batch_size, length) or
            (batch_size, height, width).
            Although this function can cope with any shape `array`, with any number of dimensions.
        n_fourier_features: Total number of requested Fourier features. Must be an even number
            because half a sine and half are cosine.
        min_freq: If min_freq=2 and array is in the range [0, 1] then the lowest freq "wave" will
            go from -1 to +1 across the dimension.
        max_freq:

    Returns:
        fourier_features: An np.ndarray of the same dtype as `array`,
            with shape `array.shape + (n_fourier_features,)`. Fourier features with even indexes
            are sine. Odd indexes are cosine.
    """
    assert n_fourier_features % 2 == 0
    assert min_freq > 0
    assert max_freq > min_freq

    div_term = np.linspace(
        start=min_freq,
        stop=max_freq,
        num=n_fourier_features // 2,
        dtype=array.dtype,
    )
    fourier_features = np.full(
        shape=array.shape + (n_fourier_features,),
        fill_value=np.NaN,
        dtype=array.dtype,
    )

    radians = array * np.pi / 2
    radians = np.expand_dims(radians, axis=-1)
    radians_x_div_term = radians * div_term
    fourier_features[..., 1::2] = np.cos(radians_x_div_term)
    fourier_features[..., 0::2] = np.sin(radians_x_div_term)
    return fourier_features


def _rescale_coords_for_all_dims_to_approx_0_to_1(
    np_batch: NumpyBatch,
    lengths: dict[str, Number],
) -> dict[str, np.ndarray]:
    """Rescale coords for all dimensions, across all modalities.

    Args:
        lengths: The approximate lengths of each dimension across an example. For example,
            if the dimension is x_osgb then the length will be the approximate distance in meters
            from the left to the right side of an average example. If the dimension is time_utc
            then the length is the approximate total duration in seconds of the example.
            The keys must be x_osgb, y_osgb, or time_utc.
            We need to use constant lengths across all examples so the spatial encoding
            is always proportional to the "real world" length across all examples.
            If we didn't do that, a spatial encoding of 1 would represent different "real world"
            distances across examples. This would almost certainly be harmful, especially
            because we're expecting the model to learn to do some basic geometry!
    """
    rescaled_coords: dict[str, np.ndarray] = {}
    for dim_name in ("x_osgb", "y_osgb", "time_utc"):
        coords_for_dim_from_all_modalities: dict[BatchKey, np.ndarray] = {
            key: value.astype(np.float32)
            for key, value in np_batch.items()
            if key.name.endswith(dim_name)
        }
        length = lengths[dim_name]
        rescaled_coords_for_dim = _rescale_coords_for_single_dim_to_approx_0_to_1(
            coords_for_dim_from_all_modalities=coords_for_dim_from_all_modalities, length=length
        )
        rescaled_coords.update(rescaled_coords_for_dim)
    return rescaled_coords


def _rescale_coords_for_single_dim_to_approx_0_to_1(
    coords_for_dim_from_all_modalities: dict[BatchKey, np.ndarray],
    length: Number,
) -> dict[str, np.ndarray]:
    """Rescale the coords for a single dimension, across all modalities.

    Args:
        length: The approximate length of the dimension across an example. For example,
            if the dimension is x_osgb then the length will be the distance in meters
            from the left to the right side of the example. Must be positive.

    Returns:
        Dictionary where the keys are "<BatchKey.name>_rescaled", and the values
        are a numpy array of the rescaled coords. The minimum value is guaranteed to be
        0 or larger. The maximum value will depend on the length.
    """
    length = np.float32(length)
    assert length > 0
    min_per_example = _get_min_per_example(coords_for_dim_from_all_modalities)
    rescaled_arrays: dict[str, np.ndarray] = {}
    for key, array in coords_for_dim_from_all_modalities.items():
        if "satellite" in key.name and "time" not in key.name:
            # Handle 2-dimensional OSGB coords on the satellite imagery
            assert (
                len(array.shape) == 3
            ), f"Expected satellite coord to have 3 dims, not {len(array.shape)} {key.name=}"
            _min_per_example = np.expand_dims(min_per_example, axis=-1)
        else:
            _min_per_example = min_per_example

        rescaled_array = (array - _min_per_example) / length
        rescaled_arrays[f"{key.name}_rescaled"] = rescaled_array

    return rescaled_arrays


def _get_min_per_example(
    coords_for_dim_from_all_modalities: dict[BatchKey, np.ndarray]
) -> np.ndarray:
    n_modalities = len(coords_for_dim_from_all_modalities)
    assert n_modalities > 0
    batch_size = list(coords_for_dim_from_all_modalities.values())[0].shape[0]

    # Pre-allocate arrays to hold min values for each example of each modality's coordinates.
    mins = np.full((batch_size, n_modalities), fill_value=np.NaN, dtype=np.float32)

    for modality_i, coord_data_array in enumerate(coords_for_dim_from_all_modalities.values()):
        coord_data_array = coord_data_array.reshape((batch_size, -1))
        mins[:, modality_i] = np.nanmin(coord_data_array, axis=1)

    min_per_example = mins.min(axis=1)
    assert min_per_example.shape[0] == batch_size
    return np.expand_dims(min_per_example, axis=-1)
