import json
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.earthworm import Client as EW_Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth
from obspy import Stream
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, windows, convolve
from scipy.fftpack import next_fast_len
from collections import OrderedDict
from xarray import DataArray
from grid_utils import calculate_time_buffer
import fnmatch
import warnings
from warning_config import RTMWarning


plt.ioff()  # Don't show the figure unless fig.show() is explicitly called

# Load AVO infrasound station calibration values (units are Pa/ct)
AVO_INFRA_CALIB_FILE = 'avo_infra_calib_vals.json'
with open(AVO_INFRA_CALIB_FILE) as f:
    avo_calib_values = json.load(f)

# Load AVO infrasound station coordinates (elevation units are meters)
AVO_INFRA_COORD_FILE = 'avo_infra_coords.json'
with open(AVO_INFRA_COORD_FILE) as f:
    avo_coords = json.load(f)

# Define IRIS and AVO clients (define WATC client within function)
iris_client = FDSN_Client('IRIS')
avo_client = EW_Client('pubavo1.wr.usgs.gov', port=16023)  # 16023 is long-term

# Channels to use in data requests - covering all the bases here!
CHANNELS = 'BDF,BDG,BDH,BDI,BDJ,BDK,HDF,DDF'

KM2M = 1000     # [m/km]
SEC2MIN = 1/60  # [min/s]


def gather_waveforms(source, network, station, starttime, endtime, buffer=0,
                     remove_response=False, return_failed_stations=False,
                     watc_username=None, watc_password=None):
    """
    Gather infrasound waveforms from IRIS or WATC FDSN, or AVO Winston, and
    output a Stream object with station/element coordinates attached.
    Optionally remove the sensitivity.

    NOTE:
        Usual RTM usage is to specify a starttime/endtime that brackets the
        estimated source origin time. Then buffer is used to download enough
        extra data to account for the time required for an infrasound signal to
        propagate to the farthest station. Because this buffer is so critical,
        this function issues a warning if it remains set to its default of 0 s.

    Args:
        source: Which source to gather waveforms from - options are:
                'IRIS' <-- IRIS FDSN
                'WATC' <-- WATC FDSN
                'AVO'  <-- AVO Winston
        network: SEED network code
        station: SEED station code
        starttime: Start time for data request (UTCDateTime)
        endtime: End time for data request (UTCDateTime)
        buffer: [s] Extra amount of data to download after endtime (default: 0)
        remove_response: Toggle conversion to Pa via remove_sensitivity() if
                         available, else just do a simple scalar multiplication
                         (default: False)
        return_failed_stations: If True, returns a list of station codes that
                                were requested but not downloaded. This
                                disables the standard failed station warning
                                message (default: False)
        watc_username: Username for WATC FDSN server (default: None)
        watc_password: Password for WATC FDSN server (default: None)
    Returns:
        st_out: Stream containing gathered waveforms
        failed_stations: (Optional) List containing station codes that were
                         requested but not downloaded
    """

    print('--------------')
    print('GATHERING DATA')
    print('--------------')

    # Warn if buffer is set to 0 s
    if buffer == 0:
        warnings.warn('Buffer is set to 0 seconds. Are you sure you\'ve '
                      'downloaded enough data for RTM?', RTMWarning)

    # IRIS FDSN
    if source == 'IRIS':

        print('Reading data from IRIS FDSN...')
        try:
            st_out = iris_client.get_waveforms(network, station, '*', CHANNELS,
                                               starttime, endtime + buffer,
                                               attach_response=remove_response)
        except FDSNNoDataException:
            st_out = Stream()  # Just create an empty Stream object

    # WATC FDSN
    elif source == 'WATC':

        print('Connecting to WATC FDSN...')
        watc_client = FDSN_Client('http://10.30.5.10:8080',
                                  user=watc_username,
                                  password=watc_password)

        print('Successfully connected. Reading data from WATC FDSN...')
        try:
            st_out = watc_client.get_waveforms(network, station, '*', CHANNELS,
                                               starttime, endtime + buffer,
                                               attach_response=remove_response)
        except FDSNNoDataException:
            st_out = Stream()  # Just create an empty Stream object

    # AVO Winston
    elif source == 'AVO':

        print('Reading data from AVO Winston...')
        try:
            # Array case
            if station in ['ADKI', 'AKS', 'DLL', 'OKIF', 'SDPI']:

                # Select the correct channel
                if station in ['DLL', 'OKIF']:
                    channel = 'HDF'
                else:
                    channel = 'BDF'

                st_out = Stream()  # Make an empty Stream object to populate

                # Deal with funky channel naming convention for AKS (for all
                # other arrays, six numbered elements are assumed)
                if station == 'AKS':
                    for channel in ['BDF', 'BDG', 'BDH', 'BDI', 'BDJ', 'BDK']:
                        st_out += avo_client.get_waveforms(network, station,
                                                           '--', channel,
                                                           starttime,
                                                           endtime + buffer)
                else:
                    for location in ['01', '02', '03', '04', '05', '06']:
                        st_out += avo_client.get_waveforms(network, station,
                                                           location, channel,
                                                           starttime,
                                                           endtime + buffer)

            # Single station case
            else:
                st_out = avo_client.get_waveforms(network, station, '--',
                                                  'BDF', starttime,
                                                  endtime + buffer)

                # Special case for CLES1 and CLES2 which also have HDF channels
                if station in ['CLES1', 'CLES2']:
                    st_out += avo_client.get_waveforms(network, station, '--',
                                                       'HDF', starttime,
                                                       endtime + buffer)

        # KeyError means that the station is not on AVO Winston for ANY time
        # period, OR that the user didn't format the request (e.g., station
        # string) appropriately
        except KeyError:
            st_out = Stream()  # Just create an empty Stream object

    else:
        raise ValueError('Unrecognized source. Valid options are \'IRIS\', '
                         '\'WATC\', or \'AVO\'.')

    st_out.sort()

    # Check that all requested stations are present in Stream
    requested_stations = station.split(',')
    downloaded_stations = [tr.stats.station for tr in st_out]
    failed_stations = []
    for sta in requested_stations:
        # The below check works with wildcards, but obviously cannot detect if
        # ALL stations corresponding to a given wildcard (e.g., O??K) were
        # downloaded. Thus, if careful station selection is desired, specify
        # each station explicitly and the below check will then be effective.
        if not fnmatch.filter(downloaded_stations, sta):
            if not return_failed_stations:
                # If we're not returning the failed stations, then show this
                # warning message to alert the user
                warnings.warn(f'Station {sta} not downloaded from {source} '
                              'server for this time period.', RTMWarning)
            failed_stations.append(sta)

    # If the Stream is empty, then we can stop here
    if st_out.count() == 0:
        print('No data downloaded.')
        if return_failed_stations:
            return st_out, failed_stations
        else:
            return st_out

    # Otherwise, show what the Stream contains
    print(st_out.__str__(extended=True))  # This syntax prints the WHOLE Stream

    # Add zeros to ensure all Traces have same length
    st_out.trim(starttime, endtime + buffer, pad=True, fill_value=0)

    print('Assigning coordinates...')

    # Assign coordinates using IRIS FDSN regardless of data source
    try:
        inv = iris_client.get_stations(network=network, station=station,
                                       starttime=starttime,
                                       endtime=endtime + buffer,
                                       level='channel')
    except FDSNNoDataException:
        inv = []

    for tr in st_out:
        for nw in inv:
            for sta in nw:
                for cha in sta:
                    # Being very thorough to check if everything matches!
                    if (tr.stats.network == nw.code and
                            tr.stats.station == sta.code and
                            tr.stats.location == cha.location_code and
                            tr.stats.channel == cha.code):

                        tr.stats.longitude = cha.longitude
                        tr.stats.latitude = cha.latitude
                        tr.stats.elevation = cha.elevation

    # Check if any Trace did NOT get coordinates assigned, and try to use JSON
    # coordinates if available
    for tr in st_out:
        try:
            tr.stats.longitude, tr.stats.latitude, tr.stats.elevation
        except AttributeError:
            try:
                tr.stats.latitude, tr.stats.longitude,\
                    tr.stats.elevation = avo_coords[tr.stats.station]
                warnings.warn(f'Using coordinates from JSON file for {tr.id}.',
                              RTMWarning)
            except KeyError:
                print(f'No coordinates available for {tr.id}. Stopping.')
                raise

    # Remove sensitivity
    if remove_response:

        print('Removing sensitivity...')

        for tr in st_out:
            try:
                # Just removing sensitivity for now. remove_response() can lead
                # to errors. This should be sufficient for now. Plus some
                # IRIS-AVO responses are wonky.
                tr.remove_sensitivity()
            except ValueError:
                try:
                    calib = avo_calib_values[tr.stats.station]
                    tr.data = tr.data * calib
                    tr.stats.processing.append('RTM: Data multiplied by '
                                               f'calibration value of {calib} '
                                               'Pa/ct')
                    warnings.warn('Using calibration value from JSON file for '
                                  f'{tr.id}.', RTMWarning)
                except KeyError:
                    print('No calibration value available for {tr.id}. '
                          'Stopping.')
                    raise

    print('Done')

    # Return the Stream with coordinates attached (and responses removed if
    # specified)
    if return_failed_stations:
        return st_out, failed_stations
    else:
        return st_out


def gather_waveforms_bulk(lon_0, lat_0, max_radius, starttime, endtime,
                          buffer=0, remove_response=False, watc_username=None,
                          watc_password=None):
    """
    Bulk gather infrasound waveforms within a specified maximum radius of a
    specified location. Waveforms are gathered from IRIS (and optionally WATC)
    FDSN, and AVO Winston. Outputs a Stream object with station/element
    coordinates attached. Optionally removes the sensitivity. [Output Stream
    has the same properties as output Stream from gather_waveforms().]

    NOTE 1:
        WATC database will NOT be used for station search NOR data download
        unless BOTH watc_username and watc_password are set.

    NOTE 2:
        Usual RTM usage is to specify a starttime/endtime that brackets the
        estimated source origin time. Then buffer is used to download enough
        extra data to account for the time required for an infrasound signal to
        propagate to the farthest station. This function can automatically
        calculate an appropriate buffer amount (it assumes that the station
        search center and source grid center are identical, which in practice
        should be the case since the grid center should be used as the station
        search center).

    Args:
        lon_0: [deg] Longitude of search center
        lat_0: [deg] Latitude of search center
        max_radius: [km] Maximum radius to search for stations within
        starttime: Start time for data request (UTCDateTime)
        endtime: End time for data request (UTCDateTime)
        buffer: Either a buffer time in s or an RTM grid (i.e., an
                xarray.DataArray output from define_grid() for this event). If
                a grid is specified, the buffer time in s is automatically
                calculated based upon the grid params and this function's
                station locations. This is the extra amount of data to download
                after endtime, and is simply passed on to the calls to
                gather_waveforms() (default: 0)
        remove_response: Toggle conversion to Pa via remove_sensitivity() if
                         available, else just do a simple scalar multiplication
                         (default: False)
        watc_username: Username for WATC FDSN server (default: None)
        watc_password: Password for WATC FDSN server (default: None)
    Returns:
        st_out: Stream containing bulk gathered waveforms
    """

    print('-------------------')
    print('BULK GATHERING DATA')
    print('-------------------')

    print('Creating station list...')

    # Grab IRIS inventory - not accounting for buffer here
    iris_inv = iris_client.get_stations(starttime=starttime, endtime=endtime,
                                        channel=CHANNELS, level='channel')

    inventories = [iris_inv]  # Add IRIS inventory to list

    # If the user supplied both a WATC password and WATC username, then search
    # through WATC database
    if watc_username and watc_password:

        print('Connecting to WATC FDSN...')
        watc_client = FDSN_Client('http://10.30.5.10:8080', user=watc_username,
                                  password=watc_password)
        print('Successfully connected.')

        # Grab WATC inventory - not accounting for buffer here
        watc_inv = watc_client.get_stations(starttime=starttime,
                                            endtime=endtime, channel=CHANNELS,
                                            level='channel')

        inventories.append(watc_inv)  # Add WATC inventory to list

    requested_station_list = []  # Initialize list of stations to request

    max_station_dist = 0  # [m] Keep track of the most distant station

    # Big loop through all channels in all inventories!
    for inv in inventories:
        for nw in inv:
            for stn in nw:
                for cha in stn:
                    dist, _, _ = gps2dist_azimuth(lat_0, lon_0, cha.latitude,
                                                  cha.longitude)  # [m]
                    if dist <= max_radius * KM2M:
                        requested_station_list.append(stn.code)
                        # Keep track of most distant station (within radius)
                        if dist > max_station_dist:
                            max_station_dist = dist

    # Loop through each entry in AVO infrasound station coordinates JSON file
    for sta, coord in avo_coords.items():
        dist, _, _ = gps2dist_azimuth(lat_0, lon_0, *coord[0:2])  # [m]
        if dist <= max_radius * KM2M:
            requested_station_list.append(sta)
            # Keep track of most distant station (within radius)
            if dist > max_station_dist:
                max_station_dist = dist

    if not requested_station_list:
        raise ValueError('Station list is empty. Expand the station search '
                         'and try again.')

    # Put into the correct format for ObsPy (e.g., 'HOM,O22K,DLL')
    requested_stations = ','.join(np.unique(requested_station_list))

    print('Done')

    # Check if buffer is an xarray.DataArray - if so, the user wants a buffer
    # time to be automatically calculated from this grid
    if type(buffer) == DataArray:
        buffer = calculate_time_buffer(grid=buffer,
                                       max_station_dist=max_station_dist)  # [s]

    if buffer != 0:
        print(f'Using buffer of {buffer:.1f} s (~{buffer * SEC2MIN:.0f} min)')

    print('Making calls to gather_waveforms()...')

    st_out = Stream()  # Initialize empty Stream to populate

    # Gather waveforms from IRIS
    iris_st, iris_failed = gather_waveforms(source='IRIS', network='*',
                                            station=requested_stations,
                                            starttime=starttime,
                                            endtime=endtime, buffer=buffer,
                                            remove_response=remove_response,
                                            return_failed_stations=True)
    st_out += iris_st

    # If IRIS couldn't grab all stations in requested station list, try WATC
    # (if the user set username and password)
    if iris_failed:

        if watc_username and watc_password:
            # Gather waveforms from WATC
            watc_st, watc_failed = gather_waveforms(source='WATC', network='*',
                                                    station=','.join(iris_failed),
                                                    starttime=starttime,
                                                    endtime=endtime,
                                                    buffer=buffer,
                                                    remove_response=remove_response,
                                                    return_failed_stations=True,
                                                    watc_username=watc_username,
                                                    watc_password=watc_password)
        else:
            # Return an empty Stream and same failed stations
            watc_st, watc_failed = Stream(), iris_failed

        st_out += watc_st

        # If WATC couldn't grab all stations missed by IRIS, try AVO
        if watc_failed:

            # Gather waveforms from AVO
            remaining_failed = []
            for sta in watc_failed:
                avo_st, avo_failed = gather_waveforms(source='AVO',
                                                      network='AV',
                                                      station=sta,
                                                      starttime=starttime,
                                                      endtime=endtime,
                                                      buffer=buffer,
                                                      remove_response=remove_response,
                                                      return_failed_stations=True)

                st_out += avo_st
                remaining_failed += avo_failed

            if remaining_failed:
                print('--------------')
                for sta in remaining_failed:
                    warnings.warn(f'Station {sta} found in radius search but '
                                  'no data found.', RTMWarning)

    print('--------------')
    print('Finishing gathering waveforms from station list. Check warnings '
          'above for any missed stations.')

    return st_out


def process_waveforms(st, freqmin, freqmax, envelope=False,
                      decimation_rate=None, smooth_win=None, agc_params=None,
                      normalize=False, plot_steps=False):
    """
    Process infrasound waveforms. By default, the input Stream is detrended,
    tapered, and filtered. Optional: Enveloping, decimation (via
    interpolation), automatic gain control (AGC), and normalization. If no
    decimation rate is specified, Traces are simply interpolated to the lowest
    sample rate present in the Stream. Optionally plots the Stream after each
    processing step has been applied for troubleshooting.

    Args:
        st: Stream from gather_waveforms()
        freqmin: [Hz] Lower corner for zero-phase bandpass filter
        freqmax: [Hz] Upper corner for zero-phase bandpass filter
        envelope: Take envelope of waveforms (default: False)
        decimation_rate: [Hz] New sample rate to decimate to (via
                         interpolation). If None, just interpolates to the
                         lowest sample rate present in the Stream (default:
                         None)
        smooth_win: [s] Smoothing window duration. If None, does not perform
                    smoothing (default: None)
        agc_params: Dictionary of keyword arguments to be passed on to _agc().
                    Example: dict(win_sec=500, method='gismo')
                    If set to None, no AGC is applied. For details, see the
                    docstring for _agc() (default: None)
        normalize: Apply normalization to Stream (default: False)
        plot_steps: Toggle plotting each processing step (default: False)
    Returns:
        st_out: Stream containing processed waveforms
    """

    print('---------------')
    print('PROCESSING DATA')
    print('---------------')

    print('Detrending...')
    st_d = st.copy()
    st_d.detrend(type='linear')

    print('Tapering...')
    st_t = st_d.copy()
    st_t.taper(max_percentage=0.05)

    print('Filtering...')
    st_f = st_t.copy()
    st_f.filter(type='bandpass', freqmin=freqmin, freqmax=freqmax,
                zerophase=True)

    # Gather default processed Streams into dictionary
    streams = OrderedDict(input=st,
                          detrended=st_d,
                          tapered=st_t,
                          filtered=st_f)

    if envelope:
        print('Enveloping...')
        st_e = st_f.copy()  # Copy filtered Stream from previous step
        for tr in st_e:
            npts = tr.count()
            # The below line is much faster than using obspy.signal.envelope()
            # See https://github.com/scipy/scipy/issues/6324#issuecomment-425752155
            tr.data = np.abs(hilbert(tr.data, N=next_fast_len(npts))[:npts])
            tr.stats.processing.append('RTM: Enveloped via np.abs(hilbert())')
        streams['enveloped'] = st_e

    # The below step is mandatory - either we decimate or simply equalize fs
    st_i = list(streams.values())[-1].copy()  # Copy the "newest" Stream
    a_param = 20  # This is required for the 'lanczos' method; may affect speed
    if decimation_rate:
        print('Decimating...')
        st_i.interpolate(sampling_rate=decimation_rate, method='lanczos',
                         a=a_param)
        streams['decimated'] = st_i
    else:
        print('Equalizing sampling rates...')
        lowest_fs = np.min([tr.stats.sampling_rate for tr in st_i])
        st_i.interpolate(sampling_rate=lowest_fs, method='lanczos', a=a_param)
        streams['sample_rate_equalized'] = st_i
    # After this step, all Traces in the Stream have the same sampling rate!

    if smooth_win:
        print('Smoothing...')
        st_s = st_i.copy()  # Copy interpolated Stream from previous step
        # Calculate number of samples to use in window
        smooth_win_samp = int(st_s[0].stats.sampling_rate * smooth_win)
        if smooth_win_samp < 1:
            raise ValueError('Smoothing window too short.')
        win = windows.hann(smooth_win_samp)  # Use Hann window
        for tr in st_s:
            tr.data = convolve(tr.data, win, mode='same') / sum(win)
            tr.stats.processing.append(f'RTM: Smoothed with {smooth_win} s '
                                       'Hann window')
        streams['smoothed'] = st_s

    if agc_params:
        print('Applying AGC...')
        # Using the "newest" Stream below (copied within the AGC function)
        st_a = _agc(list(streams.values())[-1], **agc_params)
        streams['agc'] = st_a

    if normalize:
        print('Normalizing...')
        st_n = list(streams.values())[-1].copy()  # Copy the "newest" Stream
        st_n.normalize()
        streams['normalized'] = st_n

    print('Done')

    if plot_steps:
        print('Generating processing plots...')
        for title, st in streams.items():
            fig = plt.figure(figsize=(8, 8))
            st.plot(fig=fig, equal_scale=False)
            fig.axes[0].set_title(title)
            fig.canvas.draw()
            fig.tight_layout()
            fig.show()
        print('Done')

    st_out = list(streams.values())[-1]  # Final entry in Stream dictionary

    return st_out


def _agc(st, win_sec, method='gismo'):
    """
    Apply automatic gain correction (AGC) to a collection of waveforms stored
    in an ObsPy Stream object. This function is designed to be used as part of
    process_waveforms() though it can be used on its own as well.

    Args:
        st: Stream containing waveforms to be processed
        win_sec: [s] AGC window. A shorter time window results in a more
                     aggressive AGC effect (i.e., increased gain for quieter
                     signals)
        method: One of 'gismo' or 'walker' (default: 'gismo')

                'gismo' A Python implementation of agc.m from the GISMO suite:
                            https://github.com/geoscience-community-codes/GISMO/blob/master/core/%40correlation/agc.m
                        It preserves the relative amplitudes of traces (i.e.
                        doesn't normalize) but is limited in how much in can
                        boost quiet sections of waveform.

               'walker' An implementation of the AGC algorithm described in
                        Walker et al. (2010), paragraph 22:
                            https://doi.org/10.1029/2010JB007863
                        (The code is adopted from Richard Sanderson's version.)
                        This method scales the amplitudes of the resulting
                        traces between [-1, 1] (or [0, 1] for envelopes) so
                        inter-trace amplitudes are not preserved. However, the
                        method produces a stronger AGC effect which may be
                        desirable depending upon the context.
    Returns:
        st_out: Copy of input Stream with AGC applied
    """

    st_out = st.copy()

    if method == 'gismo':

        for tr in st_out:

            win_samp = int(tr.stats.sampling_rate * win_sec)

            scale = np.zeros(tr.count() - 2 * win_samp)
            for i in range(-1 * win_samp, win_samp + 1):
                scale = scale + np.abs(tr.data[win_samp + i:
                                               win_samp + i + scale.size])

            scale = scale / scale.mean()  # Using max() here may better
                                          # preserve inter-trace amplitudes

            # Fill out the ends of scale with its first/last values
            scale = np.hstack((np.ones(win_samp) * scale[0],
                               scale,
                               np.ones(win_samp) * scale[-1]))

            tr.data = tr.data / scale  # "Scale" the data, sample-by-sample

            tr.stats.processing.append('RTM: AGC applied via '
                                       f'_agc(win_sec={win_sec}, '
                                       f'method=\'{method}\')')

    elif method == 'walker':

        for tr in st_out:

            half_win_samp = int(tr.stats.sampling_rate * win_sec / 2)

            scale = []
            for i in range(half_win_samp, tr.count() - half_win_samp):
                # The window is centered on index i
                scale_max = np.abs(tr.data[i - half_win_samp:
                                           i + half_win_samp]).max()
                scale.append(scale_max)

            # Fill out the ends of scale with its first/last values
            scale = np.hstack((np.ones(half_win_samp) * scale[0],
                               scale,
                               np.ones(half_win_samp) * scale[-1]))

            tr.data = tr.data / scale  # "Scale" the data, sample-by-sample

            tr.stats.processing.append('RTM: AGC applied via '
                                       f'_agc(win_sec={win_sec}, '
                                       f'method=\'{method}\')')

    else:
        raise ValueError(f'AGC method \'{method}\' not recognized. Method '
                         'must be either \'gismo\' or \'walker\'.')

    return st_out
