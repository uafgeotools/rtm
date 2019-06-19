import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
from obspy.geodetics import gps2dist_azimuth
import utm
import time
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
        attrs = dict(UTM=dict(zone=zone_number, southern_hemisphere=lat_0 < 0))
    else:
        attrs = dict(UTM=None)
    grid_out = xr.DataArray(data, coords=[('y', y), ('x', x)], attrs=attrs)

    # Plot grid preview, if specified
    if plot:
        if projected:
            proj = ccrs.UTM(**grid_out.attrs['UTM'])
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


def grid_search(processed_st, grid, celerity_list, stack_method='sum'):
    """
    Perform a grid search over x, y, and celerity (c) and return a 4-D object
    with dimensions x, y, t, and c. Also return time-shifted Streams for each
    (x, y, c) point.

    Args:
        processed_st: Pre-processed Stream <-- output of process_waveforms()
        grid: x, y grid to use <-- output of define_grid()
        celerity_list: List of celerities to use
        stack_method: Method to use for stacking aligned waveforms. One of
                      'sum' or 'product' (default: 'sum')

                'sum' Sum the aligned waveforms sample-by-sample.

            'product' Multiply the aligned waveforms sample-by-sample. Results
                      in a more spatially concentrated stack maximum.
    Returns:
        S: xarray.DataArray object containing the 4-D (t, c, y, x) stack
           function
        shifted_streams: NumPy array with dimensions (c, y, x) containing the
                         time-shifted Streams
    """

    # Define global time axis using the first Trace of the input Stream
    times = processed_st[0].times(type='utcdatetime')

    # Expand grid dimensions in celerity and time
    S = grid.expand_dims(dict(celerity=np.float64(celerity_list))).copy()
    S = S.expand_dims(dict(time=times.astype('datetime64[ns]'))).copy()

    # Pre-allocate NumPy array to store Streams for each grid point
    shifted_streams = np.empty(shape=S.shape[1:], dtype=object)

    total_its = np.product(S.shape[1:])  # Don't count time dimension
    counter = 0
    tic = time.process_time()
    for i, celerity in enumerate(S['celerity'].values):

        for j, y_coord in enumerate(S['y']):

            for k, x_coord in enumerate(S['x']):

                st = processed_st.copy()

                for tr in st:

                    if grid.attrs['UTM']:
                        grid_zone_number = grid.attrs['UTM']['zone']
                        *station_utm, _, _ = utm.from_latlon(tr.stats.latitude,
                                                             tr.stats.longitude,
                                                             force_zone_number=grid_zone_number)

                        # Check if station is outside of grid UTM zone
                        _, _, station_zone_number, _ = utm.from_latlon(tr.stats.latitude,
                                                                       tr.stats.longitude)
                        if station_zone_number != grid_zone_number:
                            warnings.warn(f'{tr.id} locates to UTM zone '
                                          f'{station_zone_number} instead of grid UTM '
                                          f'zone {grid_zone_number}. Consider '
                                          'reducing station search extent or using an '
                                          'unprojected grid.')

                        # Distance is in meters
                        distance = np.linalg.norm(np.array(station_utm) -
                                                  np.array([x_coord, y_coord]))

                    else:
                        # Distance is in meters
                        distance, _, _ = gps2dist_azimuth(y_coord, x_coord,
                                                          tr.stats.latitude,
                                                          tr.stats.longitude)

                    time_shift = distance / celerity  # [s]
                    tr.stats.starttime = tr.stats.starttime - time_shift
                    tr.stats.processing.append('RTM: Shifted by '
                                               f'-{time_shift:.2f} s')

                # Trim to time limits of input Stream
                st.trim(times[0], times[-1], pad=True, fill_value=0)

                if stack_method == 'sum':
                    stack = np.sum([tr.data for tr in st], axis=0)

                elif stack_method == 'product':
                    stack = np.product([tr.data for tr in st], axis=0)

                else:
                    raise ValueError(f'Stack method \'{stack_method}\' not '
                                     'recognized. Method must be either '
                                     '\'sum\' or \'product\'.')

                # Assign stacked time series to this latitude/longitude point
                S.loc[dict(x=x_coord, y=y_coord, celerity=celerity)] = stack

                # Save the time-shifted Stream
                shifted_streams[i, j, k] = st

                # Print grid search progress
                counter += 1
                print('{:.1f}%'.format((counter / total_its) * 100))

    toc = time.process_time()
    print(f'Done (elapsed time = {toc-tic:.1f} s)')

    return S, shifted_streams
