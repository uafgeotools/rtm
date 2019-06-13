#%% (1) Grab the data

import json
from obspy import UTCDateTime
from utils import gather_waveforms, process_waveforms

# watc_credentials.json contains a single line with format ["user", "password"]
with open('watc_credentials.json') as f:
    watc_username, watc_password = json.load(f)

t1 = UTCDateTime('2016-05-22T07:45:00')
t2 = t1 + 40*60

st = gather_waveforms(source='IRIS', network='AK,TA',
                      station='HOM,M19K,M22K,O20K,O22K,RC01', starttime=t1,
                      endtime=t2, remove_response=True,
                      watc_username=watc_username, watc_password=watc_password)

#%% (2) Process the data

FREQ_MIN = 0.5        # [Hz] Lower bandpass corner
FREQ_MAX = 2          # [Hz] Upper bandpass corner

INTERP_RATE = 0.05    # [Hz] New sampling rate to interpolate to

AGC_WIN = 250         # [s] Window for AGC
AGC_METHOD = 'gismo'  # Method to use for AGC, specify 'gismo' or 'walker'

agc_params = dict(win_sec=AGC_WIN, method=AGC_METHOD)

st_proc = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, interp_rate=INTERP_RATE,
                            agc_params=agc_params, normalize=True,
                            plot_steps=True)

#%% (3) Grid search

import numpy as np

X0 = -152.9902  # [deg] Longitude of grid center
Y0 = 60.0183    # [deg] Latitude of grid center

X_RADIUS = 2    # [deg] E-W grid radius (half of grid "width")
Y_RADIUS = 2    # [deg] N-S grid radius (half of grid "height")

SPACING = 0.25  # [deg] Grid spacing


def define_grid(x0, y0, x_radius, y_radius, spacing):
    x_out, dx = np.linspace(x0 - x_radius, x0 + x_radius,
                            int(2 * x_radius / spacing) + 1, retstep=True)

    y_out, dy = np.linspace(y0 - y_radius, y0 + y_radius,
                            int(2 * y_radius / spacing) + 1, retstep=True)

    if dx != spacing:
        print(f'Warning: Requested spacing of {spacing} does not match '
              f'np.linspace()-returned x-axis spacing of {dx:.6f}.')

    if dy != spacing:
        print(f'Warning: Requested spacing of {spacing} does not match '
              f'np.linspace()-returned y-axis spacing of {dy:.6f}.')

    if not (X0 in x_out and Y0 in y_out):
        print('Warning: (x0, y0) is not located in grid. Check for numerical '
              'precision problems (i.e., rounding error).')

    return x_out, y_out


x, y = define_grid(x0=X0, y0=Y0, x_radius=X_RADIUS, y_radius=Y_RADIUS,
                   spacing=SPACING)
