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
# EXPOSE PUBLIC MODULES
# -----------------------------------------------------------------------------

from .grid import define_grid, produce_dem, grid_search, calculate_time_buffer
from .waveform import process_waveforms
from .plotting import plot_time_slice, plot_record_section, plot_st, plot_stack_peak
from .stack import get_peak_coordinates
from .travel_time import prepare_fdtd_run, fdtd_travel_time
