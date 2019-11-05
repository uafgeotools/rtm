import warnings
from obspy import UTCDateTime
import numpy as np
import utm
from scipy.signal import find_peaks
from . import RTMWarning

def get_peak_coordinates(S, global_max=True, height=None, min_time=None,
                         unproject=False):
    """
    Find the values of the coordinates corresponding to the maxima (peaks) in
    a stack function S. Function will return all peaks above the "height" and
    separated by greater than "min_time" in the stack function. Optionally
    "unprojects" UTM coordinates to (latitude, longitude) for projected grids.

    Args:
        S: xarray.DataArray containing the stack function S
        global_max: Return the values for only the max of the stack function
                    (default: True)
        height: Minimum threshold for the value of a detection (peak) in
                the stack function S (default: None). Only used if
                global_max=False.
        min_time: Minimum time (distance) between peaks in stack function S [s]
            (default: None). Only used if global_max=False.
        unproject: If True and if S is a projected grid, unprojects the UTM
                   coordinates to (latitude, longitude) (default: False)
    Returns:
        time_max: Time (UTCDateTime) corresponding to peaks in S
        y_max: [deg lat. or m N] y-coordinates corresponding to peaks in S
        x_max: [deg lon. or m E] x-coordinates corresponding to peaks in S
    """

    # Create peak stack function over time
    s_peak = S.max(axis=(1, 2)).data

    # Return just the global max or desired peaks. Check for multiple maxima
    # along each dimension and across the stack function
    if global_max:
        print('Returning just global max!')

        #get global max
        stack_maximum = S.where(S == S.max(), drop=True)

        # Warn if we have multiple maxima along any dimension
        for dim in stack_maximum.coords:
            num_dim_maxima = stack_maximum[dim].size
            if num_dim_maxima != 1:
                warnings.warn(f'Multiple maxima ({num_dim_maxima}) present in S '
                              f'along the {dim} dimension.', RTMWarning)

        # return all peaks
        peaks, props = find_peaks(s_peak, (None, None))

        # check for multiple global maxima
        max_args = np.argwhere(props['peak_heights'] ==
                               np.amax(props['peak_heights']))

        num_global_maxima = max_args.shape[0]
        if num_global_maxima > 1:
            warnings.warn(f'Multiple global maxima ({num_global_maxima}) present '
                          'in S. Using first occurrence.', RTMWarning)

        peaks = np.array(peaks[max_args])[0]
    else:

        if (height is None) or (min_time is None):
            raise ValueError('height and min_time must be supplied for ' \
                             'peak detection when global_max=False!')

        peak_dt = (np.datetime64(S['time'][1].data) -
               np.datetime64(S['time'][0].data)) / np.timedelta64(1, 's')

        peaks, props = find_peaks (s_peak, height, distance=min_time/peak_dt)

        npeaks = len(peaks)
        print(f'Found {npeaks} peaks in stack for height > {height:.1f} and '
              f'min_time > {min_time/peak_dt:.1f} s.')

    time_max = [UTCDateTime(S['time'][i].values.astype(str)) for i in peaks]
    x_max = [S.where(S[i] == S[i].max(), drop=True).squeeze()['x'].values.tolist()
             for i in peaks]
    y_max = [S.where(S[i] == S[i].max(), drop=True).squeeze()['y'].values.tolist()
             for i in peaks]

    if global_max:
        time_max = time_max[0]
        x_max = x_max[0]
        y_max = y_max[0]

    if unproject:
        # If the grid is projected
        if S.UTM:
            print('Unprojecting coordinates from UTM to (latitude, longitude).')
            for i in range(0, npeaks):
                y_max[i], x_max[i] = utm.to_latlon(x_max[i], y_max[i],
                                                   zone_number=S.UTM['zone'],
                                                   northern=not S.UTM['southern_hemisphere'])

        # If the grid is already in lat/lon
        else:
            print('unproject=True is set but coordinates are already in '
                  '(latitude, longitude). Doing nothing.')

    return time_max, y_max, x_max
