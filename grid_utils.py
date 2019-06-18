import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
import utm
import warnings


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
    if projected:
        # Create list of grid corners
        corners = dict(SW=(x.min(), y.min()),
                       NW=(x.min(), y.max()),
                       NE=(x.max(), y.max()),
                       SE=(x.max(), y.min()))
        for label, corner in corners.items():
            # "Un-project" back to latitude/longitude
            lat, lon = utm.to_latlon(*corner, zone_number, northern=lat_0 >= 0,
                                     strict=False)
            # "Re-project" to UTM
            _, _, new_zone_number, _ = utm.from_latlon(lat, lon)
            if new_zone_number != zone_number:
                warnings.warn(f'{label} grid corner locates to UTM zone '
                              f'{new_zone_number} instead of origin UTM zone '
                              f'{zone_number}. Consider reducing grid extent '
                              'or using an unprojected grid.')

    # Create grid
    data = np.full((y.size, x.size), np.nan)  # Initialize an array of NaNs
    if projected:
        # Add the projection information to the grid metadata
        attrs = dict(utm_zone_number=zone_number)
    else:
        attrs = dict(utm_zone_number=None)
    grid_out = xr.DataArray(data, coords=[('y', y), ('x', x)], attrs=attrs)

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

        # Note that trial source locations are at the CENTER of each plotted
        # grid box
        grid_out.plot.pcolormesh(ax=ax, transform=transform,
                                 edgecolor='black', add_colorbar=False)

        # Plot the center of the grid
        ax.scatter(lon_0, lat_0, color='red', transform=ccrs.Geodetic())

        fig.show()

    return grid_out


def rtm_stack(st,stack_type):
    """
    Various stacking and processing procedures for RTM

    Args:
        st: Stream of waveforms ready to be stacked
        stack_type: Stacking options are:
                'linear' <-- linear sum, sample by sample
                'product' <-- multiplication, samplbe by sample
                'lin_win'  <-- linear sum in defined time windows. If defined needs a window length. (in progress!)
        win_len: [s] window length for stack in defined windows
    Returns:
        tr: trace containing stacked data
    """

    tr=st[0].copy()    #copy first trace over for stack trace
    tr.stats.station=stack_type     #define station name as stack type?
    tr.stats.processing.append(stack_type)  #append stack type to processing

    #how do we want to fill in station stats for the trace? perhaps fill in lat/long with grid info

    if stack_type=='linear':
        print('Peforming linear stack...')

        #Progresively sum data
        for tr_tmp in st[1:]:
            tr.data=tr.data+tr_tmp.data

    elif stack_type=='product':
        print('Peforming product stack...')

        #Progresively multipy data
        for tr_tmp in st[1:]:
            tr.data=tr.data*tr_tmp.data

    print('done')
    return tr
