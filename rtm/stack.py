from obspy import UTCDateTime
import numpy as np
import utm
from scipy.signal import find_peaks
import warnings
from . import RTMWarning


def get_max_coordinates(S, unproject=False):
    """
    Find the values of the coordinates corresponding to the global maximum in
    a stack function S. Warns if multiple maxima exist along any dimension. (If
    this is the case, the first occurrence of a maximum is used.) Optionally
    "unprojects" UTM coordinates to (latitude, longitude) for projected grids.

    Args:
        S: xarray.DataArray containing the stack function S
        unproject: If True and if S is a projected grid, unprojects the UTM
                   coordinates to (latitude, longitude) (default: False)
    Returns:
        time_max: Time (UTCDateTime) corresponding to global max(S)
        y_max: [deg lat. or m N] y-coordinate corresponding to max(S)
        x_max: [deg lon. or m E] x-coordinate corresponding to max(S)
    """

    stack_maximum = S.where(S == S.max(), drop=True)

    # Warn if we have multiple maxima along any dimension
    for dim in stack_maximum.coords:
        num_dim_maxima = stack_maximum[dim].size
        if num_dim_maxima != 1:
            warnings.warn(f'Multiple maxima ({num_dim_maxima}) present in S '
                          f'along the {dim} dimension.', RTMWarning)

    # Since the where() function above returns a subset of the original S whose
    # non-maximum values are set to nan, we must ignore these values when
    # finding the coordinates of a maximum
    max_indices = np.argwhere(~np.isnan(stack_maximum.data))

    # Warn if we have multiple global maxima
    num_global_maxima = max_indices.shape[0]
    if num_global_maxima != 1:
        warnings.warn(f'Multiple global maxima ({num_global_maxima}) present '
                      'in S. Using first occurrence.', RTMWarning)

    # Using first occurrence with [0] index below
    max_coords = stack_maximum[tuple(max_indices[0])].coords

    time_max = UTCDateTime(max_coords['time'].values.astype(str))
    y_max = max_coords['y'].values.tolist()
    x_max = max_coords['x'].values.tolist()

    if unproject:
        # If the grid is projected
        if S.UTM:
            print('Unprojecting coordinates from UTM to (latitude, longitude).')
            y_max, x_max = utm.to_latlon(x_max, y_max,
                                         zone_number=S.UTM['zone'],
                                         northern=not S.UTM['southern_hemisphere'])
        # If the grid is already in lat/lon
        else:
            print('unproject=True is set but coordinates are already in '
                  '(latitude, longitude). Doing nothing.')

    return time_max, y_max, x_max

def get_peak_coordinates(S, height, min_time, global_max=True, unproject=False):
    """
    Find the values of the coordinates corresponding to the maxima (peaks) in
    a stack function S.  Optionally "unprojects" UTM coordinates to (latitude,
    longitude) for projected grids.

    Args:
        S: xarray.DataArray containing the stack function S
        height: Minimum peak height in stack function.
        min_time: Minimum time (distance) between peaks in stack function [s].
        global_max: Return the values for only the max of the stack function
                    (default: True)
        unproject: If True and if S is a projected grid, unprojects the UTM
                   coordinates to (latitude, longitude) (default: False)
    Returns:
        time_max: Time (UTCDateTime) corresponding to peaks in S
        y_max: [deg lat. or m N] y-coordinates corresponding to peaks in S
        x_max: [deg lon. or m E] x-coordinates corresponding to peaks in S
    """

    #create peak stack function over time
    s_peak = S.max(axis=(1, 2)).data

    peak_dt = (np.datetime64(S['time'][1].data) - np.datetime64(S['time'][0].data)) \
    / np.timedelta64(1, 's')

    #find peaks in stack function
    peaks, props = find_peaks(s_peak, height, distance=min_time/peak_dt)
    npeaks = len(peaks)

    print('Found %d peaks in stack for height=%.1f and min_time=%.1f s' %
          (npeaks, height, min_time/peak_dt))

    #Use just the global max. Argmax returns the first max if multiple are present
    if global_max:
        peaks = np.array([peaks[props['peak_heights'].argmax()]])
        npeaks = len(peaks)
        print('Returning just global max!')

    time_max = [UTCDateTime(S['time'][i].values.astype(str)) for i in peaks]
    x_max = [S.where(S[i] == S[i].max(), drop=True).squeeze()['x'].values.tolist() \
             for i in peaks]
    y_max = [S.where(S[i] == S[i].max(), drop=True).squeeze()['y'].values.tolist() \
             for i in peaks]

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
