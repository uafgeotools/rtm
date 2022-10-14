# -----------------------------------------------------------------------------
# WARNING CONFIG
# -----------------------------------------------------------------------------

import warnings

# Subclass UserWarning as a "uafgeotools" warning"
class RTMWarning(UserWarning):
    UAFGEOTOOLS = True

# Make warnings more consistent
warnings.simplefilter(action='always', category=RTMWarning)

# Make a custom format for "uafgeotools" warnings
def _formatwarning(message, category, *args, **kwargs):
    if hasattr(category, 'UAFGEOTOOLS'):
        msg = f'{category.__name__}: {message}\n'  # Much cleaner
    else:
        import warnings
        msg_form = warnings.WarningMessage(message, category, *args, **kwargs)
        msg = warnings._formatwarnmsg_impl(msg_form)  # Default
    return msg
warnings.formatwarning = _formatwarning

# Clean up
del _formatwarning
del warnings

# -----------------------------------------------------------------------------
# DEFINE HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# Find UTM CRS of a (lat, lon) point (see https://gis.stackexchange.com/a/423614)
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
def _estimate_utm_crs(lat, lon, datum_name='WGS 84'):
    utm_crs_list = query_utm_crs_info(
        datum_name=datum_name,
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    return CRS.from_epsg(utm_crs_list[0].code)  # Taking first entry of list here!

# -----------------------------------------------------------------------------
# EXPOSE PUBLIC MODULES
# -----------------------------------------------------------------------------

from .grid import define_grid, produce_dem, grid_search, calculate_time_buffer
from .waveform import process_waveforms
from .plotting import plot_time_slice, plot_record_section, plot_st, plot_stack_peak
from .stack import get_peak_coordinates
from .travel_time import prepare_fdtd_run, fdtd_travel_time
