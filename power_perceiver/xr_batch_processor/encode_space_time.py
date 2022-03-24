import numpy as np
import xarray as xr

from power_perceiver.data_loader.data_loader import XarrayBatch


class EncodeSpaceTime:
    """Encode space and time in a way that Perceiver understands :)

    The broad approach is:

    For each dimension in (x, y, time):

    1. Compute the min and max across all modalities.
    2. For each modality, rescale the coordinate to [0, 1] using the min and max across all
       modalities.
    3. Compute Fourier features for each modality.

    The end result is that the XarrayBatch will now include:

    - <modality name>_position_encoding_x
    - <modality name>_position_encoding_y
    - <modality name>_position_encoding_time
    """

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        for dim_name in ("x", "y", "time"):
            #: dict keys will be of the form <modality_name>_position_encoding_<dim_name>
            coords_for_dim_from_all_modalities: dict[str, xr.DataArray] = _get_coords_for_dim(
                dim_name=dim_name, xr_batch=xr_batch
            )
            coords_for_dim_from_all_modalities = _rescale_data_arrays_to_0_to_1(
                coords_for_dim_from_all_modalities
            )
            for key, coords in coords_for_dim_from_all_modalities.items():
                xr_batch[key] = _fourier_position_encoding(coords)
        return xr_batch


def _rescale_data_arrays_to_0_to_1(data_arrays: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Rescales multiple DataArrays using the same min and max across all DataArrays.

    Args:
        data_arrays: A dictionary of xr.DataArrays.
            Each dictionary key should be of the form <modality_name>_position_encoding_<dim_name>.
            Each dictionary value should be a DataArray of shape [batch_size, ...].
            All DataArrays must have the same batch_size.

    Returns:
        rescaled_data_arrays: A dict with the same keys as the input `data_arrays` dict but where
            each DataArray has had its values rescaled to be in the range [0, 1].
    """

    # Compute the maximum and the range, across all the data_arrays.
    list_of_data_arrays = [torch.flatten(t, start_dim=1) for t in data_arrays.values()]
    data_arrays_concatenated = torch.cat(list_of_data_arrays, dim=1)
    del list_of_data_arrays
    minimum = np.nanmin(data_arrays_concatenated, axis=1)
    maximum = np.nanmax(data_arrays_concatenated, axis=1)
    min_max_range = maximum - minimum
    del maximum

    minimum = minimum.unsqueeze(-1)
    min_max_range = min_max_range.unsqueeze(-1)

    # Rescale each tensor
    rescaled_data_arrays = {}
    for name, tensor in data_arrays.items():
        if len(tensor.shape) == 3:
            # 2D OSGB coords
            rescaled_data_arrays[name] = (tensor - minimum.unsqueeze(-1)) / min_max_range.unsqueeze(
                -1
            )
        else:
            rescaled_data_arrays[name] = (tensor - minimum) / min_max_range

    return rescaled_data_arrays
