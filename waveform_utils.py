import json
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.earthworm import Client as EW_Client
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from obspy import Stream
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, windows, convolve
from scipy.fftpack import next_fast_len
from collections import OrderedDict


# Load AVO infrasound station calibration values (units are Pa/ct)
AVO_INFRA_CALIB_FILE = 'avo_infra_calib_vals.json'
with open(AVO_INFRA_CALIB_FILE) as f:
    avo_calib_values = json.load(f)

# Define IRIS and AVO clients (define WATC client within function)
iris_client = FDSN_Client('IRIS')
avo_client = EW_Client('pubavo1.wr.usgs.gov', port=16023)  # 16023 is long-term


def gather_waveforms(source, network, station, starttime, endtime,
                     remove_response=False, watc_username=None,
                     watc_password=None):
    """
    Gather infrasound waveforms from IRIS or WATC FDSN, or AVO Winston, and
    output a Stream object with station/element coordinates attached.
    Optionally remove the sensitivity.

    Args:
        source: Which source to gather waveforms from - options are:
                'IRIS' <-- IRIS FDSN
                'WATC' <-- WATC FDSN
                'AVO'  <-- AVO Winston
        network: SEED network code
        station: SEED station code
        starttime: Start time for data request (UTCDateTime)
        endtime: End time for data request (UTCDateTime)
        remove_response: Toggle conversion to Pa via remove_sensitivity() if
                         available, else just do a simple scalar multiplication
                         (default: False)
        watc_username: Username for WATC FDSN server (default: None)
        watc_password: Password for WATC FDSN server (default: None)
    Returns:
        st_out: Stream containing gathered waveforms
    """

    print('--------------')
    print('GATHERING DATA')
    print('--------------')

    # IRIS FDSN
    if source == 'IRIS':

        print('Reading data from IRIS FDSN...')
        st_out = iris_client.get_waveforms(network, station, '*', 'BDF,HDF',
                                           starttime, endtime,
                                           attach_response=remove_response)

    # WATC FDSN
    elif source == 'WATC':

        print('Connecting to WATC FDSN...')
        try:
            watc_client = FDSN_Client('http://10.30.5.10:8080',
                                      user=watc_username,
                                      password=watc_password)
        except FDSNException:
            print('Issue connecting to WATC FDSN. Check your VPN '
                  'connection and try again.')
            return Stream()

        print('Successfully connected. Reading data from WATC FDSN...')
        st_out = watc_client.get_waveforms(network, station, '*', 'BDF,HDF',
                                           starttime, endtime,
                                           attach_response=remove_response)

    # AVO Winston
    elif source == 'AVO':

        print('Reading data from AVO Winston...')

        # Array case
        if station in ['ADKI', 'AKS', 'DLL', 'OKIF', 'SDPI']:

            # Select the correct channel
            if station in ['DLL', 'OKIF']:
                channel = 'HDF'
            else:
                channel = 'BDF'

            st_out = Stream()  # Make an empty Stream object to populate

            # Deal with funky channel naming convention for AKS (for all other
            # arrays, six numbered elements are assumed)
            if station == 'AKS':
                for channel in ['BDF', 'BDG', 'BDH', 'BDI', 'BDJ', 'BDK']:
                    st_out += avo_client.get_waveforms(network, station, '--',
                                                       channel, starttime,
                                                       endtime)
            else:
                for location in ['01', '02', '03', '04', '05', '06']:
                    st_out += avo_client.get_waveforms(network, station,
                                                       location, channel,
                                                       starttime, endtime)

        # Single station case
        else:
            st_out = avo_client.get_waveforms(network, station, '--', 'BDF',
                                              starttime, endtime)

            # Special case for CLES1 and CLES2 which also have HDF channels
            if station in ['CLES1', 'CLES2']:
                st_out += avo_client.get_waveforms(network, station, '--',
                                                   'HDF', starttime, endtime)

    else:

        print('Unrecognized source. Valid options are \'IRIS\', \'WATC\', or '
              '\'AVO\'.')
        return Stream()

    # Add zeros to ensure all Traces have same length
    st_out.trim(starttime, endtime, pad=True, fill_value=0)

    st_out.sort()

    print(st_out)

    print('---------------------')
    print('ASSIGNING COORDINATES')
    print('---------------------')

    # Assign coordinates using IRIS FDSN regardless of data source
    try:
        inv = iris_client.get_stations(network=network, station=station,
                                       starttime=starttime, endtime=endtime,
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

    # Report if any Trace did NOT get coordinates assigned
    print('Traces WITHOUT coordinates assigned:')
    num_unassigned = 0
    for tr in st_out:
        try:
            tr.stats.longitude, tr.stats.latitude
        except AttributeError:
            print('\t' + tr.id)
            num_unassigned += 1
    if num_unassigned == 0:
        print('\tNone')

    # Remove sensitivity
    if remove_response:

        print('--------------------')
        print('REMOVING SENSITIVITY')
        print('--------------------')

        unremoved_ids = []
        for tr in st_out:
            print(tr.id)
            try:
                # Just removing sensitivity for now. remove_response() can lead
                # to errors. This should be sufficient for now. Plus some
                # IRIS-AVO responses are wonky.
                tr.remove_sensitivity()
                print('\tSensitivity removed using attached response.')
            except ValueError:
                print('\tNo response information available.')
                try:
                    calib = avo_calib_values[tr.stats.station]
                    tr.data = tr.data * calib
                    tr.stats.processing.append('RTM: Data multiplied by '
                                               f'calibration value of {calib} '
                                               'Pa/ct')
                    print('\tSensitivity removed using calibration value of '
                          f'{calib} Pa/ct.')
                except KeyError:
                    print('\tNo calibration value available.')
                    unremoved_ids.append(tr.id)

        # Report if any Trace did NOT get sensitivity removed
        print('Traces WITHOUT sensitivity removed:')
        [print('\t' + tr_id) for tr_id in unremoved_ids]
        if len(unremoved_ids) == 0:
            print('\tNone')

    return st_out


def process_waveforms(st, freqmin, freqmax, envelope=False, interp_rate=None,
                      smooth_win=None, agc_params=None, normalize=False,
                      plot_steps=False):
    """
    Process infrasound waveforms. By default, the input Stream is detrended,
    tapered, and filtered. Optional: Enveloping, interpolation (decimation),
    automatic gain control (AGC), and normalization. Optionally plots the
    Stream after each processing step has been applied for troubleshooting.

    Args:
        st: Stream from gather_waveforms()
        freqmin: [Hz] Lower corner for zero-phase bandpass filter
        freqmax: [Hz] Upper corner for zero-phase bandpass filter
        envelope: Take envelope of waveforms (default: False)
        interp_rate: [Hz] New sample rate to interpolate to. If None, does not
                     perform interpolation (default: None)
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
        st_e = list(streams.values())[-1].copy()  # Copy the "newest" Stream
        for tr in st_e:
            npts = tr.count()
            # The below line is much faster than using obspy.signal.envelope()
            # See https://github.com/scipy/scipy/issues/6324#issuecomment-425752155
            tr.data = np.abs(hilbert(tr.data, N=next_fast_len(npts))[:npts])
            tr.stats.processing.append('RTM: Enveloped via np.abs(hilbert())')
        streams['enveloped'] = st_e

    if interp_rate:
        print('Interpolating...')
        st_i = list(streams.values())[-1].copy()  # Copy the "newest" Stream
        st_i.interpolate(sampling_rate=interp_rate, method='lanczos', a=20)
        streams['interpolated'] = st_i

    if smooth_win:
        print('Smoothing...')
        st_s = list(streams.values())[-1].copy()  # Copy the "newest" Stream
        for tr in st_s:
            # Calculate number of samples to use in window
            smooth_win_samp = int(tr.stats.sampling_rate * smooth_win)
            if smooth_win_samp < 1:
                raise ValueError('Smoothing window too short.')
            win = windows.hann(smooth_win_samp)  # Use Hann window
            tr.data = convolve(tr.data, win, mode='same') / sum(win)
            tr.stats.processing.append(f'RTM: Smoothed with {smooth_win} s '
                                       'Hann window')
        streams['smoothed'] = st_s

    if agc_params:
        print('Applying AGC...')
        # Using the "newest" Stream below (copied within the AGC function)
        st_a = _agc(list(streams.values())[-1].copy(), **agc_params)
        streams['agc'] = st_a

    if normalize:
        print('Normalizing...')
        st_n = list(streams.values())[-1].copy()  # Copy the "newest" Stream
        st_n.normalize()
        streams['normalized'] = st_n

    print('\tDone')

    if plot_steps:
        print('Generating processing plots...')
        for title, st in streams.items():
            fig = plt.figure(figsize=(8, 8))
            st.plot(fig=fig, equal_scale=False)
            fig.axes[0].set_title(title)
            fig.tight_layout()
            fig.show()
        print('\tDone')

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

        st_out.detrend('demean')  # NOTE: Causes envelopes to go negative...

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
