from obspy import UTCDateTime
import numpy as np
import utm
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
        celerity_max: [m/s] Celerity corresponding to global max(S)
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
    celerity_max = max_coords['celerity'].values.tolist()
    y_max = max_coords['y'].values.tolist()
    x_max = max_coords['x'].values.tolist()

    if unproject:
        # If the grid is projected
        if S.attrs['UTM']:
            print('Unprojecting coordinates from UTM to (latitude, longitude).')
            y_max, x_max = utm.to_latlon(x_max, y_max,
                                         zone_number=S.attrs['UTM']['zone'],
                                         northern=not S.attrs['UTM']['southern_hemisphere'])
        # If the grid is already in lat/lon
        else:
            print('unproject=True is set but coordinates are already in '
                  '(latitude, longitude). Doing nothing.')

    return time_max, celerity_max, y_max, x_max
