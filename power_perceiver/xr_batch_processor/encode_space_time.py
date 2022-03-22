class EncodeSpaceTime:
    """Encode space and time in a way that Perceiver understands :)

    The broad approach is:

    For each dimension (x, y, time):

    1. Compute the min and max across all modalities.
    2. For each modality, rescale the coordinate to [0, 1], using the min and max across all
       modalities.
    3. Compute Fourier features for each modality.

    The end result is that the XarrayBatch will now include:

    - <modality name>_position_encoding_x
    - <modality name>_position_encoding_y
    - <modality name>_position_encoding_time

    e.g.:

    - pv_position_encoding_x
    - hrvsatellite_position_encoding_time
    """

    pass
