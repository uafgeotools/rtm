#%% (1) Define grid

from rtm import define_grid, produce_dem

"""
To obtain the below file from OpenTopography, run the command

    $ curl https://cloud.sdsc.edu/v1/AUTH_opentopography/hosted_data/OTDS.072019.4326.1/raster/DEM_WGS84.tif -o DEM_WGS84.tif

or simply paste the above URL in a web browser. Alternatively, specify None to
automatically download and use 1 arc-second STRM data.
"""
EXTERNAL_FILE = 'DEM_WGS84.tif'

LON_0 = 169.448212  # [deg] Longitude of grid center
LAT_0 = -19.527908  # [deg] Latitude of grid center

X_RADIUS = 600  # [m] E-W grid radius (half of grid "width")
Y_RADIUS = 650  # [m] N-S grid radius (half of grid "height")
SPACING = 10    # [m] Grid spacing

grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=True)

dem = produce_dem(grid, external_file=EXTERNAL_FILE)

#%% (2) Grab and process the data

from obspy import UTCDateTime
from waveform_collection import gather_waveforms
from rtm import process_waveforms, calculate_time_buffer

# Start and end of time window containing (suspected) events
STARTTIME = UTCDateTime('2016-07-30T05:22:45')
ENDTIME = STARTTIME + 10

# Data collection parameters
SOURCE = 'IRIS'
NETWORK = '3E'
STATION = 'YIF?'
LOCATION = '*'
CHANNEL = '*'

MAX_STATION_DIST = 0.8  # [km] Max. dist. from grid center to station (approx.)

FREQ_MIN = 0.5  # [Hz] Lower bandpass corner
FREQ_MAX = 10   # [Hz] Upper bandpass corner

DECIMATION_RATE = 10  # [Hz] New sampling rate to use for decimation
SMOOTH_WIN = 1        # [s] Smoothing window duration

# Automatically determine appropriate time buffer in s
time_buffer = calculate_time_buffer(grid, MAX_STATION_DIST)

st = gather_waveforms(source=SOURCE, network=NETWORK, station=STATION,
                      location=LOCATION, channel=CHANNEL, starttime=STARTTIME,
                      endtime=ENDTIME, time_buffer=time_buffer)

st_proc = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, smooth_win=SMOOTH_WIN,
                            decimation_rate=DECIMATION_RATE, normalize=True)

#%% (3) Perform grid search

from rtm import grid_search

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'
TIME_METHOD = 'celerity'  # Choose either 'celerity' or 'fdtd'
TIME_KWARGS = {'celerity': 343, 'dem': dem}

S = grid_search(processed_st=st_proc, grid=grid, time_method=TIME_METHOD,
                stack_method=STACK_METHOD, **TIME_KWARGS)

#%% (4) Plot

from rtm import (plot_time_slice, plot_record_section, get_peak_coordinates,
                 plot_stack_peak, plot_st)

fig_st = plot_st(st, filt=[FREQ_MIN, FREQ_MAX], equal_scale=False, rem_resp=True,
                 label_waveforms=True)

fig_peak = plot_stack_peak(S, plot_max=True)

fig_slice = plot_time_slice(S, st_proc, label_stations=True, dem=dem)

time_max, y_max, x_max, _, _ = get_peak_coordinates(S, global_max=False,
                                                    height=3, min_time=2,
                                                    unproject=S.UTM)

fig = plot_record_section(st_proc, origin_time=time_max[0],
                          source_location=(y_max[0], x_max[0]),
                          plot_celerity=S.celerity, label_waveforms=True)
