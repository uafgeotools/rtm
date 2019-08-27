#%% (1) Define grid

from rtm import define_grid

LON_0 = -152.238  # [deg] Longitude of grid center
LAT_0 = 61.296    # [deg] Latitude of grid center

PROJECTED = False

if PROJECTED:
    X_RADIUS = 250  # [m] E-W grid radius (half of grid "width")
    Y_RADIUS = 250  # [m] N-S grid radius (half of grid "height")
    SPACING = 50    # [m] Grid spacing

else:
    X_RADIUS = 1   # [deg] E-W grid radius (half of grid "width")
    Y_RADIUS = 1   # [deg] N-S grid radius (half of grid "height")
    SPACING = 0.1  # [deg] Grid spacing

grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=PROJECTED,
                   plot_preview=False)

#%% (2) Grab and process the data

import json
from obspy import UTCDateTime
from rtm import gather_waveforms_bulk, process_waveforms

# Start and end of time window containing (suspected) events
STARTTIME = UTCDateTime('2019-07-15T16:10')
ENDTIME = STARTTIME + 60*60

MAX_RADIUS = 500        # [km] Radius within which to search for stations

FREQ_MIN = 0.5          # [Hz] Lower bandpass corner
FREQ_MAX = 2            # [Hz] Upper bandpass corner

DECIMATION_RATE = 0.1   # [Hz] New sampling rate to use for decimation

SMOOTH_WIN = 60         # [s] Smoothing window duration

# watc_credentials.json contains a single line with format ["user", "password"]
with open('watc_credentials.json') as f:
    watc_username, watc_password = json.load(f)

st = gather_waveforms_bulk(LON_0, LAT_0, MAX_RADIUS, STARTTIME, ENDTIME,
                           time_buffer=grid, remove_response=True,
                           watc_username=watc_username,
                           watc_password=watc_password)

st_proc = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, smooth_win=SMOOTH_WIN,
                            decimation_rate=DECIMATION_RATE, agc_params=None,
                            normalize=True, plot_steps=False)

#%% (3) Perform grid search

from rtm import grid_search

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'

CELERITY = 295  # [m/s]

S, shifted_streams = grid_search(processed_st=st_proc, grid=grid,
                                 celerity=CELERITY, starttime=STARTTIME,
                                 endtime=ENDTIME, stack_method=STACK_METHOD)

#%% (4) Plot

from rtm import plot_time_slice, plot_record_section, get_max_coordinates

fig = plot_time_slice(S, st_proc, time_slice=None, label_stations=True,
                      hires=True)

time_max, y_max, x_max = get_max_coordinates(S, unproject=S.UTM)

fig = plot_record_section(st_proc, origin_time=time_max,
                          source_location=(y_max, x_max),
                          plot_celerity=[S.celerity], label_waveforms=True)

#%% DEM sandbox

from rtm import define_grid, produce_dem

EXTERNAL_FILE = 'DEM_Union_UAV_161116_sm101.tif'

grid = define_grid(lon_0=169.447, lat_0=-19.532, x_radius=5000, y_radius=5000,
                   spacing=5, projected=True)

dem = produce_dem(grid, external_file=EXTERNAL_FILE)
