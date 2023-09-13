"""
Test the three scenarios presented in Fee et al. (2021) using infresnel instead of
infraFDTD to compute travel times.
"""
import matplotlib.pyplot as plt

from rtm.travel_time import infresnel_travel_time
from waveform_collection import load_json_file
from pathlib import Path
import xarray as xr
from pyproj import CRS
from obspy.clients.fdsn import Client
from obspy import Trace, Stream
import colorcet
import matplotlib
from rtm import _proj_from_grid

SCENARIO = 'shishaldin'  # One of 'yasur', 'sakurajima', or 'shishaldin'

# Load in the celerities (from Table 1 in Fee et al. (2021))
celerities = load_json_file('test_data/celerities.json')

# Initialize FDSN client for grabbing station locations (hacky but faster than manual?)
client = Client('iris')

# Pick the right parameters and input files for this scenario
celerity = celerities[SCENARIO]
scenario_dir = Path('test_data') / SCENARIO
dem_file = sorted(scenario_dir.glob('*.tif'))[0]  # Only 1 of these in each folder
fdtd_file = sorted(scenario_dir.glob('*.nc'))[0]  # "                            "

# Load in DEM (just to get the CRS?) — this works because ALL DEMs are in UTM!!!!!!!
dem = xr.open_dataarray(dem_file).squeeze()
dem_crs = CRS(dem.rio.crs)
zone = int(dem_crs.utm_zone[:-1])
southern_hemisphere = dem_crs.utm_zone[-1] == 'S'

# Load in the infraFDTD travel times
tt_fdtd = xr.open_dataarray(fdtd_file)

# Update the Trace IDs for Yasur (special case!)
if SCENARIO == 'yasur':
    tt_fdtd = tt_fdtd.assign_coords(
        station=[tr_id.replace('YS.', '3E.').replace('.1.', '.01.').replace('.DDF', '.CDF') for tr_id in tt_fdtd.station.values]
    )

# Form grid for input (same dims as the infraFDTD grid, which is very handy!)
grid = tt_fdtd[0].drop('station')
grid.attrs['UTM'] = dict(zone=zone, southern_hemisphere=southern_hemisphere)

# Make dummy Stream that contains the proper IDs and the proper coordinates
st = Stream()
for tr_id in tt_fdtd.station.values:
    net, sta, loc, cha = tr_id.split('.')
    inv = client.get_stations(
        network=net, station=sta, channel=cha
    )
    assert len(inv) == 1 and len(inv[0]) == 1
    st += Trace(header=dict(
        network=net,
        station=sta,
        location=loc,
        channel=cha,
        latitude=inv[0][0].latitude,
        longitude=inv[0][0].longitude,
    ))

# KEY CHECK
assert set(tt_fdtd.station.values) == set([tr.id for tr in st])

#%% Option 1 — run infresnel!

sta_ind = 1  # Pick a station
tt_infresnel = infresnel_travel_time(grid, Stream(st[sta_ind]), celerity=celerity, dem_file=dem_file)
if False:
    del tt_infresnel.attrs['UTM']
    tt_infresnel.to_netcdf(scenario_dir / f'{SCENARIO}_infresnel.nc')

#%% Option 2 — load from existing file

sta_ind = 1  # Pick a station
tt_infresnel = xr.open_dataarray(scenario_dir / f'{SCENARIO}_infresnel.nc')

#%% Plot

# -------------------------
# Universal plotting params
# -------------------------
BIN_LIMITS = (-0.5, 3.5)  # [s] For histogram
CLIM = (0, 0.7)  # [s] For colormap
CMAP = plt.get_cmap('cet_fire_r')
# -------------------------

# Some definitions
diff = tt_fdtd[sta_ind] - tt_infresnel[0]  # [s]
diff_label = 'Travel time difference (s), $infraFDTD - infresnel$'
sta_x, sta_y = _proj_from_grid(grid).transform(
    st[sta_ind].stats.latitude, st[sta_ind].stats.longitude
)

# Histogram
fig, ax = plt.subplots()
_, bins, patches = ax.hist(diff.values.flatten(), bins=500, range=BIN_LIMITS)
norm = plt.Normalize(*CLIM)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for bin_center, patch in zip(bin_centers, patches):
    patch.set_facecolor(CMAP(norm(bin_center)))
ax.set_xlabel(diff_label)
ax.set_ylabel('Counts')
ax.set_xlim(BIN_LIMITS)
fig.show()

# Map
fig, ax_diff = plt.subplots()
diff.plot.imshow(
    vmin=CLIM[0], vmax=CLIM[1], cmap=CMAP, ax=ax_diff, cbar_kwargs=dict(label=diff_label)
)
ax_diff.set_aspect('equal')
ax_diff.set_xlabel('UTM easting (m)')
ax_diff.set_ylabel('UTM northing (m)')
ax_diff.ticklabel_format(style='plain')
ax_diff.scatter(sta_x, sta_y, color='black')
fig.tight_layout()
fig.show()

# DEM
fig, ax = plt.subplots()
hs = dem.copy()
hs.data = matplotlib.colors.LightSource().hillshade(
    dem.data,
    dx=abs(dem.x.diff('x').mean().values),
    dy=abs(dem.y.diff('y').mean().values),
)
hs.plot.imshow(cmap='cet_gray', ax=ax)
ax.set_aspect('equal')
ax.set_xlim(ax_diff.get_xlim())
ax.set_ylim(ax_diff.get_ylim())
ax.set_xlabel('UTM easting (m)')
ax.set_ylabel('UTM northing (m)')
ax.ticklabel_format(style='plain')
ax.set_title('DEM hillshade')
ax.scatter(sta_x, sta_y, color='black')
fig.tight_layout()
fig.axes[-1].remove()  # Remove cbar without changing the size of the map
fig.show()
