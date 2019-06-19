#%% (1) Grab and process the data

from obspy import UTCDateTime
import json
from waveform_utils import gather_waveforms, process_waveforms

# Start and end of time window containing (suspected) events
STARTTIME = UTCDateTime('2016-05-22T07:45:00')
ENDTIME = STARTTIME + 30*60

FREQ_MIN = 0.5          # [Hz] Lower bandpass corner
FREQ_MAX = 2            # [Hz] Upper bandpass corner

DECIMATION_RATE = 0.05  # [Hz] New sampling rate to use for decimation

SMOOTH_WIN = 120        # [s] Smoothing window duration

AGC_WIN = 250           # [s] AGC window duration
AGC_METHOD = 'gismo'    # Method to use for AGC, specify 'gismo' or 'walker'

# watc_credentials.json contains a single line with format ["user", "password"]
with open('watc_credentials.json') as f:
    watc_username, watc_password = json.load(f)

st = gather_waveforms(source='IRIS', network='AK,TA',
                      station='HOM,M19K,M22K,O20K,O22K,RC01',
                      starttime=STARTTIME, endtime=ENDTIME,
                      remove_response=True, watc_username=watc_username,
                      watc_password=watc_password)

agc_params = dict(win_sec=AGC_WIN, method=AGC_METHOD)

st_proc = process_waveforms(st=st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, smooth_win=SMOOTH_WIN,
                            decimation_rate=DECIMATION_RATE, agc_params=None,
                            normalize=True, plot_steps=False)

#%% (2) Define grid and perform grid search

from grid_utils import define_grid, grid_search

LON_0 = -152.9902  # [deg] Longitude of grid center
LAT_0 = 60.0183    # [deg] Latitude of grid center

PROJECTED = True

if PROJECTED:
    X_RADIUS = 50000  # [m] E-W grid radius (half of grid "width")
    Y_RADIUS = 50000  # [m] N-S grid radius (half of grid "height")
    SPACING = 5000    # [m] Grid spacing

else:
    X_RADIUS = 5   # [deg] E-W grid radius (half of grid "width")
    Y_RADIUS = 5   # [deg] N-S grid radius (half of grid "height")
    SPACING = 0.5  # [deg] Grid spacing

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'

CELERITY_LIST = [300, 310, 320]  # [m/s]

grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=PROJECTED,
                   plot_preview=False)

S, shifted_streams = grid_search(processed_st=st_proc, grid=grid,
                                 celerity_list=CELERITY_LIST,
                                 stack_method=STACK_METHOD)

#%% (3) Plot

from plotting_utils import plot_time_slice

fig = plot_time_slice(S, st_proc, time_slice=None, celerity_slice=None)

fig.show()
