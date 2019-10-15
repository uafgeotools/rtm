#%% (1) Define search and source grids

from rtm import define_grid, produce_dem

# Change to DEM from https://opentopography.org/ at some point
EXTERNAL_FILE = 'DEM_Union_UAV_161116_sm101.tif'

LON_0 = 169.448212  # [deg] Longitude of grid center
LAT_0 = -19.527908   # [deg] Latitude of grid center

X_RADIUS = 500  # [m] E-W grid radius (half of grid "width")
Y_RADIUS = 500  # [m] N-S grid radius (half of grid "height")
SPACING = 10    # [m] Grid spacing

X_RADIUS_SOURCE = 1000  # [m] E-W grid radius (half of grid "width")
Y_RADIUS_SOURCE = 1000  # [m] N-S grid radius (half of grid "height")
SPACING_SOURCE = 10    # [m] Grid spacing

search_grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                          y_radius=Y_RADIUS, spacing=SPACING, projected=True)

search_dem = produce_dem(search_grid, external_file=EXTERNAL_FILE)

source_grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS_SOURCE,
                          y_radius=Y_RADIUS_SOURCE, spacing=SPACING_SOURCE,
                          projected=True)

source_dem = produce_dem(source_grid, external_file=EXTERNAL_FILE)


#%% (2) Grab and process the data

from obspy import UTCDateTime, Stream
from waveform_collection import gather_waveforms
from rtm import process_waveforms

# Start and end of time window containing (suspected) events
STARTTIME = UTCDateTime('2016-07-30T05:22:48')
ENDTIME = STARTTIME + 10

SOURCE = 'IRIS'
NETWORK = '3E'
STATION = 'YIF?'
LOCATION = '*'
CHANNEL = '*'

FREQ_MIN = .5          # [Hz] Lower bandpass corner
FREQ_MAX = 10            # [Hz] Upper bandpass corner

DECIMATION_RATE = 10  # [Hz] New sampling rate to use for decimation
SMOOTH_WIN = 1        # [s] Smoothing window duration

st = gather_waveforms(source=SOURCE, network=NETWORK, station=STATION,
                      location=LOCATION, channel=CHANNEL, starttime=STARTTIME,
                      endtime=ENDTIME)

st_proc = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=True, smooth_win=SMOOTH_WIN,
                            decimation_rate=DECIMATION_RATE, agc_params=None,
                            normalize=True, plot_steps=False)

#%% (3) Perform grid search

from rtm import grid_search

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'
TIME_METHOD = 'celerity'  # Choose either 'celerity' or 'fdtd'
TIME_KWARGS = {'celerity': 343, 'dem': search_dem}

Scel = grid_search(processed_st=st_proc, grid=search_grid,
                   time_method=TIME_METHOD, starttime=None, endtime=None,
                   stack_method=STACK_METHOD, **TIME_KWARGS)

#%% (4) Plot

from rtm import plot_time_slice, plot_record_section, get_max_coordinates

plot_time_slice(Scel, st_proc, label_stations=True, dem=source_dem)

time_max, y_max, x_max = get_max_coordinates(Scel, unproject=Scel.UTM)

plot_record_section(st_proc, origin_time=time_max,
                    source_location=(y_max, x_max),
                    plot_celerity=Scel.celerity, label_waveforms=True)
