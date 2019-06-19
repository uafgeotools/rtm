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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
import numpy as np

# Get coordinates of peak
max_coords = S.where(S == S.max(), drop=True).squeeze()#[0][0][0]
t_max = max_coords['time'].values
c_max = max_coords['celerity'].values
y_max = max_coords['y'].values
x_max = max_coords['x'].values

if S.attrs['UTM']:
    proj = ccrs.UTM(**S.attrs['UTM'])
    transform = proj
else:
    # This is a good projection to use since it preserves area
    proj = ccrs.AlbersEqualArea(central_longitude=LON_0,
                                central_latitude=LAT_0,
                                standard_parallels=(S['y'].values.min(),
                                                    S['y'].values.max()))
    transform = ccrs.PlateCarree()

fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw=dict(projection=proj))

# Since projected grids cover less area and may not include coastlines,
# use a background image to provide geographical context (can be slow)
if S.attrs['UTM']:
    zoom_level = 8
    ax.add_image(Stamen(style='terrain-background'), zoom_level)

# Since unprojected grids have regional/global extent, just show the
# coastlines
else:
    scale = '50m'
    feature = cfeature.LAND.with_scale(scale)
    ax.add_feature(feature, facecolor=cfeature.COLORS['land'],
                   edgecolor='black')
    ax.background_patch.set_facecolor(cfeature.COLORS['water'])

S.sel(time=t_max, celerity=c_max).plot.pcolormesh(ax=ax, alpha=0.5,
                                                  transform=transform)

# Plot center of grid
ax.scatter(LON_0, LAT_0, s=100, color='red', marker='*',
           transform=ccrs.Geodetic())

# Plot stations
for tr in st_proc:
    ax.scatter(tr.stats.longitude,  tr.stats.latitude, color='black',
               transform=ccrs.Geodetic())
    ax.text(tr.stats.longitude, tr.stats.latitude,
            '  {}.{}'.format(tr.stats.network, tr.stats.station),
            verticalalignment='center_baseline', horizontalalignment='left',
            transform=ccrs.Geodetic())

fig.show()

# Processed (input) Stream
fig = plt.figure()
st_proc.plot(fig=fig)
fig.show()

# Time-shifted (output) Stream
inds = np.argwhere(S.data == S.data.max())[0]
st = shifted_streams[tuple(inds[1:])]
fig = plt.figure()
st.plot(fig=fig)
fig.show()

# Stack function
fig, ax = plt.subplots()
S.sel(y=y_max, x=x_max, celerity=c_max).plot(ax=ax)
fig.show()
