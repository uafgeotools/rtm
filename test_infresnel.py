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

SCENARIO = 'yasur'  # One of 'yasur', 'sakurajima', or 'shishaldin'

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

sta_ind = 5  # Pick a station
tt_infresnel = infresnel_travel_time(grid, Stream(st[sta_ind]), celerity=celerity, dem_file=dem_file)
if False:
    del tt_infresnel.attrs['UTM']
    tt_infresnel.to_netcdf(scenario_dir / f'{SCENARIO}_infresnel.nc')

#%% Option 2 — load from existing file

sta_ind = 5  # Pick a station
tt_infresnel = xr.open_dataarray(scenario_dir / f'{SCENARIO}_infresnel.nc')

#%% Plot

# --------------------------------------------------------------------------------------
# Universal plotting params
# --------------------------------------------------------------------------------------
TT_LIMITS = (-0.08, 0.08)  # [s]
DEM_UNDERLAY = True  # Whether to plot the DEM underlay or make separate figure for DEM
# --------------------------------------------------------------------------------------

# Some definitions
if DEM_UNDERLAY:
    cmap = plt.get_cmap('cet_CET_D11')  # Isoluminant colormap, since we have varying brightness of hillshade!
else:
    cmap = plt.get_cmap('cet_coolwarm')  # A more aggressive diverging colormap
diff = tt_fdtd[sta_ind] - tt_infresnel[0]  # [s]
sta_x, sta_y = _proj_from_grid(grid).transform(
    st[sta_ind].stats.latitude, st[sta_ind].stats.longitude
)
alpha = 0.9 if DEM_UNDERLAY else 1  # Travel times must be translucent if underlying DEM

# Set up axes
fig, (ax_hist, cax, ax_map) = plt.subplots(
    nrows=3, gridspec_kw=dict(height_ratios=[0.2, 0.03, 1]), figsize=(6, 6)
)

# Plot travel time difference map
im = diff.plot.imshow(
    vmin=TT_LIMITS[0], vmax=TT_LIMITS[1], cmap=cmap, ax=ax_map, add_colorbar=False, add_labels=False, alpha=alpha, zorder=1
)
ax_map.set_aspect('equal')
ax_map.tick_params(top=True, right=True, which='both')
ax_map.set_xlabel(f'UTM zone {dem_crs.utm_zone} easting (m)')
ax_map.set_ylabel(f'UTM zone {dem_crs.utm_zone} northing (m)')
ax_map.ticklabel_format(style='plain')
ax_map.scatter(sta_x, sta_y, color='black')
sta_label = 3 * ' ' + '.'.join(str(diff.station.values).split('.')[:2])
ax_map.text(sta_x, sta_y, sta_label, va='center', weight='bold')

# Plot colorbar
fig.colorbar(im, cax=cax, orientation='horizontal')
cax.set_xlabel('Travel time difference (s), $infraFDTD - infresnel$')

# Plot histogram
ax_hist.sharex(cax)
_, bins, patches = ax_hist.hist(diff.values.flatten(), bins=500, range=TT_LIMITS)
norm = plt.Normalize(*TT_LIMITS)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for bin_center, patch in zip(bin_centers, patches):
    patch.set_facecolor(cmap(norm(bin_center)))
    patch.set_alpha(alpha)
ax_hist.axis('off')

# Final tweaks before adding DEM
fig.tight_layout()
fig.subplots_adjust(hspace=0)  # Seems to work as I want it to?

# Plot DEM
hs = dem.copy()
hs.data = matplotlib.colors.LightSource().hillshade(
    dem.data,
    dx=abs(dem.x.diff('x').mean().values),
    dy=abs(dem.y.diff('y').mean().values),
)
if DEM_UNDERLAY:
    # Plot into existing axes
    xlim = ax_map.get_xlim()
    ylim = ax_map.get_ylim()
    hs.plot.imshow(cmap='cet_gray', ax=ax_map, add_colorbar=False, add_labels=False, zorder=-1)
    ax_map.set_aspect('equal')
    ax_map.set_xlim(xlim)
    ax_map.set_ylim(ylim)
    fig.show()
else:
    # Make new figure
    fig, ax = plt.subplots(figsize=(6, 5))
    hs.plot.imshow(cmap='cet_gray', ax=ax, add_colorbar=False, add_labels=False)
    ax.set_aspect('equal')
    ax.set_xlim(ax_map.get_xlim())
    ax.set_ylim(ax_map.get_ylim())
    ax.tick_params(top=True, right=True, which='both')
    ax.set_xlabel(f'UTM zone {dem_crs.utm_zone} easting (m)')
    ax.set_ylabel(f'UTM zone {dem_crs.utm_zone} northing (m)')
    ax.ticklabel_format(style='plain')
    ax.scatter(sta_x, sta_y, color='black')
    ax.text(sta_x, sta_y, sta_label, va='center', weight='bold')
    fig.tight_layout()
    fig.show()
