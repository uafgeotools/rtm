#%% (1) Define grid

from rtm import define_grid

LON_0 = -152.238  # [deg] Longitude of grid center
LAT_0 = 61.296    # [deg] Latitude of grid center

X_RADIUS = 1   # [deg] E-W grid radius (half of grid "width")
Y_RADIUS = 1   # [deg] N-S grid radius (half of grid "height")
SPACING = 0.1  # [deg] Grid spacing

grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=False,
                   plot_preview=False)

#%% (2) Grab and process the data

from obspy import UTCDateTime
from waveform_collection import gather_waveforms_bulk, load_json_file, \
                                INFRASOUND_CHANNELS
from rtm import calculate_time_buffer, process_waveforms

# Start and end of time window containing (suspected) events
STARTTIME = UTCDateTime('2019-07-15T16:10')
ENDTIME = STARTTIME + 60*60

MAX_RADIUS = 500        # [km] Radius within which to search for stations

FREQ_MIN = 0.5          # [Hz] Lower bandpass corner
FREQ_MAX = 2            # [Hz] Upper bandpass corner

DECIMATION_RATE = 0.1   # [Hz] New sampling rate to use for decimation

SMOOTH_WIN = 60         # [s] Smoothing window duration

# watc_credentials.json contains a single line with format ["user", "password"]
watc_username, watc_password = load_json_file('watc_credentials.json')

time_buffer = calculate_time_buffer(grid, MAX_RADIUS)

st = gather_waveforms_bulk(LON_0, LAT_0, MAX_RADIUS, STARTTIME, ENDTIME,
                           INFRASOUND_CHANNELS, time_buffer=time_buffer,
                           remove_response=True)

st_proc = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, smooth_win=SMOOTH_WIN,
                            decimation_rate=DECIMATION_RATE, agc_params=None,
                            normalize=True, plot_steps=False)

#%% (3) Perform grid search

from rtm import grid_search

TIME_METHOD = 'celerity'  # Choose either 'celerity' or 'fdtd'

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'

CELERITY = 295  # [m/s]

S = grid_search(processed_st=st_proc, grid=grid, time_method=TIME_METHOD,
                starttime=STARTTIME, endtime=ENDTIME,
                stack_method=STACK_METHOD, celerity=CELERITY)

#%% (4) Plot

from rtm import plot_time_slice, plot_record_section, get_max_coordinates

plot_time_slice(S, st_proc, label_stations=True, hires=True)

time_max, y_max, x_max = get_max_coordinates(S, unproject=S.UTM)

plot_record_section(st_proc, origin_time=time_max,
                    source_location=(y_max, x_max), plot_celerity=S.celerity,
                    label_waveforms=True)
