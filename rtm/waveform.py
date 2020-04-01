import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, windows, convolve
from scipy.fftpack import next_fast_len
from collections import OrderedDict


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
        st (:class:`~obspy.core.stream.Stream`): Stream from
            :func:`waveform_collection.server.gather_waveforms`
        freqmin (int or float): [Hz] Lower corner for zero-phase bandpass
            filter
        freqmax (int or float): [Hz] Upper corner for zero-phase bandpass
            filter
        envelope (bool): Take envelope of waveforms (default: `False`)
        decimation_rate (int or float): [Hz] New sample rate to decimate to
            (via interpolation). If `None`, just interpolates to the lowest
            sample rate present in the Stream (default: `None`)
        smooth_win (int or float): [s] Smoothing window duration. If `None`,
            does not perform smoothing (default: `None`)
        agc_params (dict): Dictionary of keyword arguments to be passed on to
            ``rtm.waveform._agc()``. Example: `dict(win_sec=500,
            method='gismo')`. If set to `None`, no AGC is applied. For details,
            see the docstring for ``rtm.waveform._agc()`` (default: `None`)
        normalize (bool): Apply normalization to Stream (default: `False`)
        plot_steps (bool): Toggle plotting each processing step (default:
            `False`)

    Returns:
        :class:`~obspy.core.stream.Stream` containing processed waveforms
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

    # Ensure all traces have the same number of values. Only operate on final
    # entry lof stream dictionary
    st_out = list(streams.values())[-1]

    min_starttime = np.min([tr.stats.starttime for tr in st_out])
    max_endtime = np.max([tr.stats.endtime for tr in st_out])

    # Most conservative trim possible - will zero-pad on either end
    st_out.trim(min_starttime, max_endtime, pad=True, fill_value=0)

    print('Done')

    if plot_steps:
        print('Generating processing plots...')
        for title, st in streams.items():
            fig = plt.figure(figsize=(8, 8))
            st.plot(fig=fig, equal_scale=False)
            fig.axes[0].set_title(title)
            fig.tight_layout()
            fig.show()
        print('Done')

    st_out = list(streams.values())[-1]  # Final entry in Stream dictionary

    return st_out


def _agc(st, win_sec, method='gismo'):
    """
    Apply automatic gain correction (AGC) to a collection of waveforms stored
    in an ObsPy Stream object. This function is designed to be used as part of
    :func:`process_waveforms` though it can be used on its own as well.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Stream containing waveforms to
            be processed
        win_sec (int or float): AGC window [s]. A shorter time window results
            in a more aggressive AGC effect (i.e., increased gain for quieter
            signals)
        method (str): One of `'gismo'` or `'walker'` (default: `'gismo'`)

            * `'gismo'` A Python implementation of ``agc.m`` from the GISMO
              suite:

              https://github.com/geoscience-community-codes/GISMO/blob/master/core/%40correlation/agc.m

              It preserves the relative amplitudes of traces (i.e. doesn't
              normalize) but is limited in how much in can boost quiet sections
              of waveform.

            * `'walker'` An implementation of the AGC algorithm described in
              Walker *et al.* (2010), paragraph 22:

              https://doi.org/10.1029/2010JB007863

              (The code is adopted from Richard Sanderson's version.) This
              method scales the amplitudes of the resulting traces between
              :math:`[-1, 1]` (or :math:`[0, 1]` for envelopes) so inter-trace
              amplitudes are not preserved. However, the method produces a
              stronger AGC effect which may be desirable depending upon the
              context.

    Returns:
        :class:`~obspy.core.stream.Stream`: Copy of input Stream with AGC
        applied
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
