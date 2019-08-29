"""
Reverse time migration of infrasound signals.
"""

# -----------------------------------------------------------------------------
# WARNING CONFIG
# -----------------------------------------------------------------------------

import warnings

# Subclass UserWarning
class RTMWarning(UserWarning):
    pass

# Make warnings more consistent
warnings.simplefilter(action='always', category=RTMWarning)

# Make a custom warning format for RTMWarning instances
def _formatwarning(message, category, *args, **kwargs):
    if category == RTMWarning:
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
# LOAD AVO INFRASOUND STATION DATA
# -----------------------------------------------------------------------------

import os
import json

dirname = os.path.dirname(__file__)

# Load AVO infrasound station calibration values (units are Pa/ct)
with open(os.path.join(dirname, 'avo_json', 'avo_infra_calibs.json')) as f:
    AVO_INFRA_CALIBS = json.load(f)

# Load AVO infrasound station coordinates (elevation units are meters)
with open(os.path.join(dirname, 'avo_json', 'avo_infra_coords.json')) as f:
    AVO_INFRA_COORDS = json.load(f)

# Clean up
del f
del dirname
del json
del os

# -----------------------------------------------------------------------------
# EXPOSE PUBLIC MODULES
# -----------------------------------------------------------------------------

from .grid import define_grid, produce_dem, grid_search, calculate_time_buffer
from .waveform import gather_waveforms, gather_waveforms_bulk, read_local, \
                      process_waveforms
from .plotting import plot_time_slice, plot_record_section
from .stack import get_max_coordinates
from .travel_time import prepare_fdtd_run
