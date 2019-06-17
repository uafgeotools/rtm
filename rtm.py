#%% (1) Grab the data

import json
from obspy import UTCDateTime
from waveform_utils import gather_waveforms, process_waveforms
from grid_utils import define_grid

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

INTERP_RATE = 0.05    # [Hz] New sampling rate to use for interpolation

SMOOTH_WIN = 120      # [s] Smoothing window duration

AGC_WIN = 250         # [s] AGC window duration
AGC_METHOD = 'gismo'  # Method to use for AGC, specify 'gismo' or 'walker'

agc_params = dict(win_sec=AGC_WIN, method=AGC_METHOD)

st_proc = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, smooth_win=SMOOTH_WIN,
                            interp_rate=INTERP_RATE, agc_params=agc_params,
                            normalize=True, plot_steps=True)

#%% (3) Grid search

LON_0 = -152.9902  # [deg] Longitude of grid center
LAT_0 = 60.0183    # [deg] Latitude of grid center

PROJECTED = False

PLOT = True

if PROJECTED:
    X_RADIUS = 10000  # [m] E-W grid radius (half of grid "width")
    Y_RADIUS = 10000  # [m] N-S grid radius (half of grid "height")
    SPACING = 500     # [m] Grid spacing

else:
    X_RADIUS = 5   # [deg] E-W grid radius (half of grid "width")
    Y_RADIUS = 5   # [deg] N-S grid radius (half of grid "height")
    SPACING = 0.5  # [deg] Grid spacing

grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=PROJECTED,
                   plot=PLOT)
