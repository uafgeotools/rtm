#%% (1) Define grid

from rtm import define_grid

LON_0 = 172.4  # [deg] Longitude of grid center
LAT_0 = 56.9   # [deg] Latitude of grid center

X_RADIUS = 10  # [deg] E-W grid radius (half of grid "width")
Y_RADIUS = 5   # [deg] N-S grid radius (half of grid "height")
SPACING = 1    # [deg] Grid spacing

grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=False,
                   plot_preview=True)

#%% (2) Grab and process the data

from obspy import UTCDateTime
from waveform_collection import gather_waveforms, gather_waveforms_bulk, \
                                INFRASOUND_CHANNELS
from rtm import process_waveforms

# Start and end of time window containing (suspected) events
STARTTIME = UTCDateTime('2018-12-18T23:30')
ENDTIME = UTCDateTime('2018-12-19T00:10')

FBX_LON = -147.7164  # [deg] Longitude of station search center
FBX_LAT = 64.8378    # [deg] Latitude of station search center
MAX_RADIUS = 1000    # [km] Radius within which to search for stations

FREQ_MIN = 0.5  # [Hz] Lower bandpass corner
FREQ_MAX = 2    # [Hz] Upper bandpass corner

DECIMATION_RATE = 0.1  # [Hz] New sampling rate to use for decimation
SMOOTH_WIN = 60        # [s] Smoothing window duration

TIME_BUFFER = 3*60*60  # [s] Manually defined buffer time

# Bulk waveform gather
st = gather_waveforms_bulk(FBX_LON, FBX_LAT, MAX_RADIUS, STARTTIME, ENDTIME,
                           INFRASOUND_CHANNELS, time_buffer=TIME_BUFFER,
                           remove_response=True)

# Add in AVO's Sand Point infrasound array
st += gather_waveforms('AVO', 'AV', 'SDPI', '0?', 'BDF', STARTTIME, ENDTIME,
                       time_buffer=TIME_BUFFER, remove_response=True)

st_proc = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, smooth_win=SMOOTH_WIN,
                            decimation_rate=DECIMATION_RATE, normalize=True)

#%% (3) Perform grid search

from rtm import grid_search

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'
TIME_METHOD = 'celerity'  # Choose either 'celerity' or 'fdtd'
CELERITY = 300  # [m/s]

S = grid_search(processed_st=st_proc, grid=grid, time_method=TIME_METHOD,
                starttime=STARTTIME, endtime=ENDTIME,
                stack_method=STACK_METHOD, celerity=CELERITY)

#%% (4) Plot

from rtm import plot_time_slice, plot_record_section, get_peak_coordinates

plot_time_slice(S, st_proc, label_stations=False, hires=True)

time_max, y_max, x_max = get_peak_coordinates(S, unproject=S.UTM)

fig = plot_record_section(st_proc, origin_time=time_max,
                          source_location=(y_max, x_max),
                          plot_celerity='range', label_waveforms=False)
fig.axes[0].set_ylim(bottom=1100)  # Start at this distance (km) from source
