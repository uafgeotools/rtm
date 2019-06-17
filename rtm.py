#%% (1) Grab the data

import json
from obspy import UTCDateTime
from utils import gather_waveforms, process_waveforms

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

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
import utm
import warnings


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


def define_grid(lon_0, lat_0, x_radius, y_radius, spacing, projected=False,
                plot=False):
    """
    Define the spatial grid of trial source locations. Grid can be defined in
    either a latitude/longitude or projected UTM (i.e., Cartesian) coordinate
    system.

    Args:
        lon_0: [deg] Longitude of grid center
        lat_0: [deg] Latitude of grid center
        x_radius: [deg or m] Distance from lon_0 to edge of grid (i.e., radius)
        y_radius: [deg or m] Distance from lat_0 to edge of grid (i.e., radius)
        spacing: [deg or m] Desired grid spacing
        projected: Boolean controlling whether a latitude/longitude or UTM grid
                   is used. If True, a UTM grid is used and the units of
                   x_radius, y_radius, and spacing are interpreted in meters
                   instead of in degrees (default: False)
        plot: Toggle plotting a preview of the grid for reference and
              troubleshooting (default: False)
    Returns:
        grid_out: xarray.DataArray object containing the grid coordinates and
                  metadata
    """

    # Make coordinate vectors
    if projected:
        x_0, y_0, zone_number, _ = utm.from_latlon(lat_0, lon_0)
    else:
        x_0, y_0, = lon_0, lat_0
    # Using np.linspace() in favor of np.arange() due to precision advantages
    x, dx = np.linspace(x_0 - x_radius, x_0 + x_radius,
                        int(2 * x_radius / spacing) + 1, retstep=True)
    y, dy = np.linspace(y_0 - y_radius, y_0 + y_radius,
                        int(2 * y_radius / spacing) + 1, retstep=True)

    # Basic grid checks
    if dx != spacing:
        warnings.warn(f'Requested spacing of {spacing} does not match '
                      f'np.linspace()-returned x-axis spacing of {dx}.')
    if dy != spacing:
        warnings.warn(f'Requested spacing of {spacing} does not match '
                      f'np.linspace()-returned y-axis spacing of {dy}.')
    if not (x_0 in x and y_0 in y):
        warnings.warn('(x_0, y_0) is not located in grid. Check for numerical '
                      'precision problems (i.e., rounding error).')

    # Create grid
    data = np.full((y.size, x.size), np.nan)  # Initialize an array of NaNs
    if projected:
        coords = [('northing', y), ('easting', x)]
        # Add the projection information to the grid metadata
        proj_string = f'+proj=utm +zone={zone_number} +ellps=WGS84 +units=m ' \
                      '+no_defs'
        if lat_0 < 0:
            proj_string += ' +south'
        attrs = dict(proj=proj_string)
    else:
        coords = [('lat', y), ('lon', x)]
        attrs = dict(proj=None)
    grid_out = xr.DataArray(data, coords=coords, attrs=attrs)

    # Plot grid preview, if specified
    if plot:
        if projected:
            proj = ccrs.UTM(zone=zone_number, southern_hemisphere=lat_0 < 0)
            transform = proj
        else:
            # This is a good projection to use since it preserves area
            proj = ccrs.AlbersEqualArea(central_longitude=lon_0,
                                        central_latitude=lat_0,
                                        standard_parallels=(y.min(), y.max()))
            transform = ccrs.PlateCarree()

        fig, ax = plt.subplots(figsize=(10, 10),
                               subplot_kw=dict(projection=proj))

        # Since projected grids cover less area and may not include coastlines,
        # use a background image to provide geographical context (can be slow)
        if projected:
            zoom_level = 12
            ax.add_image(Stamen(style='terrain-background'), zoom_level)

        # Since unprojected grids have regional/global extent, just show the
        # coastlines
        else:
            scale = '10m'
            feature = cfeature.LAND.with_scale(scale)
            ax.add_feature(feature, facecolor=cfeature.COLORS['land'],
                           edgecolor='black')
            ax.background_patch.set_facecolor(cfeature.COLORS['water'])

        # Note that trial source locations are at the CENTER of each plotted
        # grid box
        grid_out.plot.pcolormesh(ax=ax, transform=transform,
                                 edgecolor='black', add_colorbar=False)

        # Plot the center of the grid
        ax.scatter(lon_0, lat_0, color='red', transform=ccrs.Geodetic())

        fig.show()

    return grid_out


grid = define_grid(lon_0=LON_0, lat_0=LAT_0, x_radius=X_RADIUS,
                   y_radius=Y_RADIUS, spacing=SPACING, projected=PROJECTED,
                   plot=PLOT)
