import numpy as np

from power_perceiver.geospatial import lat_lon_to_osgb, osgb_to_lat_lon


def test_lat_lon_to_osgb_and_visa_versa():  # noqa: D103
    lat = 50
    lon = 0

    osgb_x_out, osgb_y_out = lat_lon_to_osgb(latitude=lat, longitude=lon)
    lat_out, lon_out = osgb_to_lat_lon(x=osgb_x_out, y=osgb_y_out)

    assert np.isclose(lat_out, lat, atol=1e-7)
    assert np.isclose(lon_out, lon, atol=1e-7)
