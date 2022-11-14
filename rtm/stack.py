import warnings

import numpy as np
from obspy import UTCDateTime
from obspy.core import Stream
from scipy.signal import find_peaks

from . import RTMWarning, _proj_from_grid


def get_peak_coordinates(S, global_max=True, height=None, min_time=None,
                         prominence=None, unproject=False):
    """
    Find the values of the coordinates corresponding to the maxima (peaks) in
    a stack function :math:`S`. Function will return all peaks above the
    `height` and separated by greater than `min_time` in the stack function.
    Returns just global max if there are less than three time segments.
    Optionally "unprojects" UTM coordinates to (latitude, longitude) for
    projected grids.

    Args:
        S (:class:`~xarray.DataArray`): The stack function :math:`S`
        global_max (bool): Only return values for the max of the stack function
            (default: `True`)
        height (int or float): Minimum threshold for the value of a detection
            (peak) in :math:`S` (default: `None`). Only used if
            `global_max=False`.
        min_time (int or float): Minimum time (distance) between peaks in
            :math:`S` [s] (default: `None`). Only used if `global_max=False`.
        prominence (int or float): Minimum peak prominence. Represents the
            vertical distance between the peak and its lowest contour line.
            Only used if `global_max=False`.
        unproject (bool): If `True` and if `S` is a projected grid, unprojects
            the UTM coordinates to (latitude, longitude) (default: `False`)

    Returns:
        tuple: Tuple containing:

        - **time_max** (:class:`~obspy.core.utcdatetime.UTCDateTime`) – Time(s)
          corresponding to peak(s) in :math:`S`
        - **y_max** – [deg lat. or m N] :math:`y`-coordinate(s) corresponding
          to peak(s) in :math:`S`
        - **x_max** – [deg lon. or m E] :math:`x`-coordinate(s) corresponding
          to peak(s) in :math:`S`
        - **peaks** (:class:`numpy.ndarray`) – Peak indices
        - **props** – Dictionary containing peak properties
    """

    # Create peak stack function over time
    s_peak = S.max(axis=(1, 2)).data

    # If there are less than three values, use global_max as find_peaks fails
    if len(s_peak) < 3:
        print('Stack function contains < 3 time samples, using global_max!')
        global_max = True
        s_peak = np.hstack((0, 0, s_peak, 0))

    # Return just the global max or desired peaks. Check for multiple maxima
    # along each dimension and across the stack function
    if global_max:
        print('Returning just global max!')

        # Get global max
        stack_maximum = S.where(S == S.max(), drop=True)

        # Warn if we have multiple maxima along any dimension
        for dim in stack_maximum.coords:
            num_dim_maxima = stack_maximum[dim].size
            if num_dim_maxima != 1:
                warnings.warn(f'Multiple maxima ({num_dim_maxima}) present in S '
                              f'along the {dim} dimension.', RTMWarning)

        max_indices = np.argwhere(~np.isnan(stack_maximum.data))
        num_global_maxima = max_indices.shape[0]

        if num_global_maxima != 1:
            warnings.warn(f'Multiple global maxima ({num_global_maxima}) present '
                          'in S. Using first occurrence.', RTMWarning)

        # Find time index and values of first occurence
        first_max = np.where(stack_maximum[tuple(max_indices[0])]['time'] == S['time'])[0]
        peaks = np.array(first_max)
        props = {'peak_heights': stack_maximum[tuple(max_indices[0])].data}
        npeaks = len(peaks)

        time_max = [UTCDateTime(stack_maximum[tuple(max_indices[0])]['time'].values.astype(str))]
        x_max = [stack_maximum[tuple(max_indices[0])]['x'].values.tolist()]
        y_max = [stack_maximum[tuple(max_indices[0])]['y'].values.tolist()]

    else:

        if (height is None) or (min_time is None) or (prominence is None):
            raise ValueError('height, min_time, and prominence must be supplied '
                             'for peak detection when global_max=False!')

        # [s] Time sampling interval of S
        peak_dt = (S.time.data[1] - S.time.data[0]) / np.timedelta64(1, 's')

        # Find all peaks based on set thresholds
        peaks, props = find_peaks(s_peak, height, distance=min_time/peak_dt,
                                  prominence=prominence)

        npeaks = len(peaks)
        print(f'Found {npeaks} peaks in stack for height > {height:.1f}, '
              f'min_time > {min_time:.1f} s, and prominence > {prominence}.')

        time_max = [UTCDateTime(S['time'][i].values.astype(str)) for i in peaks]
        x_max = [S[i].where(S[i] == S[i].max(),
                         drop=True)['x'].values[0].tolist() for i in peaks]
        y_max = [S[i].where(S[i] == S[i].max(),
                         drop=True)['y'].values[0].tolist() for i in peaks]

    if unproject:
        # If the grid is projected
        if S.UTM:
            print('Unprojecting coordinates from UTM to (latitude, longitude).')

            # Convert UTM coordinates to lat/lon
            proj = _proj_from_grid(S)
            for i in range(0, npeaks):
                y_max[i], x_max[i] = proj.transform(x_max[i], y_max[i], direction='INVERSE')

        # If the grid is already in lat/lon
        else:
            print('unproject=True is set but coordinates are already in '
                  '(latitude, longitude). Doing nothing.')

    # Return just a single float for global_max
    if global_max:
        time_max = time_max[0]
        x_max = x_max[0]
        y_max = y_max[0]

    return time_max, y_max, x_max, peaks, props


def calculate_semblance(data_in):
    """
    Calculates the semblance, a measure of multi-channel coherence, following
    the definition of Neidell & Taner (1971). Assumes data are already
    time-shifted to construct the beam.

    Args:
        data_in: Time-shifted :class:`~obspy.core.stream.Stream` or
            time-shifted :class:`numpy.ndarray`

    Returns:
        :class:`numpy.ndarray`: Multi-channel coherence, defined on
        :math:`[0, 1]`
    """

    if isinstance(data_in, Stream):
        # check that all traces have the same length
        if len(set([len(tr) for tr in data_in])) != 1:
            raise ValueError('Traces in stream must have same length!')

        n = len(data_in)

        beam = np.sum([tr.data for tr in data_in], axis=0) / n
        beampower = n * np.sum(beam**2)

        avg_power = np.sum(np.sum([tr.data**2 for tr in data_in], axis=0))

    elif isinstance(data_in, np.ndarray):
        n = data_in.shape[0]

        beam = np.sum(data_in, axis=0) / n
        beampower = n * np.sum(beam**2)

        avg_power = np.sum(np.sum([data_in**2], axis=0))

    semblance = beampower / avg_power

    return semblance
