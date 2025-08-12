import os
import time
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.io.srtm import add_shading
from obspy.geodetics import gps2dist_azimuth
from pyproj import CRS, Transformer
from rasterio.enums import Resampling

from . import RTMWarning, _estimate_utm_crs, _proj_from_grid, _grid_progress_bar
from .plotting import _plot_geographic_context
from .stack import calculate_semblance
from .travel_time import celerity_travel_time, fdtd_travel_time, infresnel_travel_time

MIN_CELERITY = 220  # [m/s] Used for travel time buffer calculation

OUTPUT_DIR = 'rtm_dem'  # Name of directory to place rtm_dem_*.tif files into
                        # (created in same dir as where function is called)

# Values in brackets below are latitude, longitude, x_radius, y_radius, spacing
TEMPLATE = 'rtm_dem_{:.4f}_{:.4f}_{}x_{}y_{}m.tif'

NODATA = -9999  # Nodata value to use for output rasters

# Define some conversion factors
KM2M = 1000     # [m/km]


def define_grid(lon_0, lat_0, x_radius, y_radius, spacing, projected=False,
                plot_preview=False):
    """
    Define the spatial grid of trial source locations. Grid can be defined in
    either a latitude/longitude or projected UTM (i.e., Cartesian) coordinate
    system.

    Args:
        lon_0 (int or float): [deg] Longitude of grid center
        lat_0 (int or float): [deg] Latitude of grid center
        x_radius (int or float): [deg or m] Distance from `lon_0` to edge of
            grid (i.e., radius)
        y_radius (int or float): [deg or m] Distance from `lat_0` to edge of
            grid (i.e., radius)
        spacing (int or float): [deg or m] Desired grid spacing
        projected (bool): Toggle whether a latitude/longitude or UTM grid is
            used. If `True`, a UTM grid is used and the units of `x_radius`,
            `y_radius`, and `spacing` are interpreted in meters instead of in
            degrees (default: `False`)
        plot_preview (bool): Toggle plotting a preview of the grid for
            reference and troubleshooting (default: `False`)

    Returns:
        :class:`~xarray.DataArray` object containing the grid coordinates and
        metadata
    """

    print('-------------')
    print('DEFINING GRID')
    print('-------------')

    # Make coordinate vectors
    if projected:
        utm_crs = _estimate_utm_crs(lat_0, lon_0)
        proj = Transformer.from_crs(utm_crs.geodetic_crs, utm_crs)
        x_0, y_0 = proj.transform(lat_0, lon_0)
        zone_number = int(utm_crs.utm_zone[:-1])
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
                      f'np.linspace()-returned x-axis spacing of {dx}.',
                      RTMWarning)
    if dy != spacing:
        warnings.warn(f'Requested spacing of {spacing} does not match '
                      f'np.linspace()-returned y-axis spacing of {dy}.',
                      RTMWarning)
    if not (x_0 in x and y_0 in y):
        warnings.warn('(x_0, y_0) is not located in grid. Check for numerical '
                      'precision problems (i.e., rounding error).', RTMWarning)
    if projected:
        # Create list of grid corners in UTM zone of grid origin
        corners = dict(SW=(x.min(), y.min()),
                       NW=(x.min(), y.max()),
                       NE=(x.max(), y.max()),
                       SE=(x.max(), y.min()))
        for label, corner in corners.items():
            # "Un-project" back to latitude/longitude
            lat, lon = proj.transform(*corner, direction='INVERSE')

            # "Re-project" to UTM
            new_zone_number = int(_estimate_utm_crs(lat, lon).utm_zone[:-1])
            if new_zone_number != zone_number:
                warnings.warn(f'{label} grid corner locates to UTM zone '
                              f'{new_zone_number} instead of origin UTM zone '
                              f'{zone_number}. Consider reducing grid extent '
                              'or using an unprojected grid.', RTMWarning)

    # Create grid
    data = np.full((y.size, x.size), np.nan)  # Initialize an array of NaNs
    # Add grid "metadata"
    attrs = dict(grid_center=(lon_0, lat_0),
                 x_radius=x_radius,
                 y_radius=y_radius,
                 spacing=spacing)
    if projected:
        # Add the projection information to the grid metadata
        attrs['UTM'] = dict(zone=zone_number, southern_hemisphere=lat_0 < 0)
    else:
        attrs['UTM'] = None
    grid_out = xr.DataArray(data, coords=[('y', y), ('x', x)], attrs=attrs)

    print('Done')

    # Plot grid preview, if specified
    if plot_preview:
        print('Generating grid preview plot...')
        if projected:
            projection = ccrs.UTM(**grid_out.UTM)
            transform = projection
        else:
            # This is a good projection to use since it preserves area
            projection = ccrs.AlbersEqualArea(central_longitude=lon_0,
                                        central_latitude=lat_0,
                                        standard_parallels=(y.min(), y.max()))
            transform = ccrs.PlateCarree()

        fig, ax = plt.subplots(figsize=(10, 10),
                               subplot_kw=dict(projection=projection))

        if not projected:
            _plot_geographic_context(ax=ax)

        # Note that trial source locations are at the CENTER of each plotted
        # grid box
        grid_out.plot.pcolormesh(ax=ax, transform=transform,
                                 edgecolor='black', add_colorbar=False)

        # Plot the center of the grid
        ax.scatter(lon_0, lat_0, s=50, color='limegreen', edgecolor='black',
                   label='Grid center', transform=ccrs.PlateCarree())

        # Add a legend
        ax.legend(loc='best')

        fig.show()

        print('Done')

    return grid_out


def produce_dem(grid, external_file=None, plot_output=True, output_file=False):
    """
    Produce a digital elevation model (DEM) with the same extent, spacing, and
    UTM projection as an input projected RTM grid. The data source can be
    either a user-supplied file or global SRTM 1 arc-second (~30 m) data taken
    from the GMT server. A DataArray and (optionally) a GeoTIFF file are
    created. Optionally plot the output DEM. Output GeoTIFF files are placed in
    ``./rtm_dem`` (relative to where this function is called).

    **NOTE 1**

    The filename convention for output files is

    ``rtm_dem_<lat_0>_<lon_0>_<x_radius>x_<y_radius>y_<spacing>m.tif``

    See the docstring for :func:`define_grid` for details/units.

    **NOTE 2**

    PyGMT caches downloaded SRTM tiles in the directory ``~/.gmt/server/earth/``.
    If you're concerned about space, you can delete this directory. It will
    be created again the next time an SRTM tile is requested.

    Args:
        grid (:class:`~xarray.DataArray`): Projected :math:`(x, y)` grid; i.e.,
            output of :func:`define_grid` with `projected=True`
        external_file (str): Filename of external DEM file to use. If `None`,
            then SRTM data is used (default: `None`)
        plot_output (bool): Toggle plotting a hillshade of the output DEM -
            useful for identifying voids or artifacts (default: `True`)
        output_file (bool): Toggle creation of output GeoTIFF file (default:
            `False`)

    Returns:
        2-D :class:`~xarray.DataArray` of elevation values with identical shape
        to input grid.
    """

    print('--------------')
    print('PROCESSING DEM')
    print('--------------')

    # Define transform to convert (lat, lon) points to a grid's UTM projection
    proj = _proj_from_grid(grid)

    # If an external DEM file was not supplied, use SRTM data
    if not external_file:

        print('No external DEM file provided. Will use 1 arc-second SRTM data '
              'from GMT server. Checking for PyGMT...')

        try:
            import pygmt
        except ImportError as error:
            raise ImportError(
                'PyGMT not found. Please install via\n\nconda install --channel conda-forge pygmt\n\nand try again.'
            ) from error

        # Find corners going clockwise from SW (in UTM coordinates)
        corners_utm = [(grid.x.values.min(), grid.y.values.min()),
                       (grid.x.values.min(), grid.y.values.max()),
                       (grid.x.values.max(), grid.y.values.max()),
                       (grid.x.values.max(), grid.y.values.min())]

        # Convert to lat/lon
        lats = []
        lons = []
        for corner in corners_utm:
            lat, lon = proj.transform(*corner, direction='INVERSE')
            lats.append(lat)
            lons.append(lon)

        # [deg] (lonmin, lonmax, latmin, latmax)
        # np.floor and np.ceil here ensure we download more data than we need
        region = [np.floor(min(lons)),
                  np.ceil(max(lons)),
                  np.floor(min(lats)),
                  np.ceil(max(lats))]

        # Use PyGMT to download DEM (or retrieve from cache)
        with pygmt.config(GMT_VERBOSE='e'):  # Suppress warnings
            dem = pygmt.datasets.load_earth_relief(
                resolution='01s', region=region, use_srtm=True
            )
        dem.rio.write_crs(dem.horizontal_datum, inplace=True)

    # If an external DEM file was supplied, use it
    else:

        dem_file = Path(str(external_file)).expanduser().resolve()

        # Check if file actually exists
        if not dem_file.is_file():
            raise FileNotFoundError(dem_file)

        print(f'Using external DEM file:\n\t{dem_file}')

        dem = xr.open_dataarray(dem_file)

    # Clean DEM before going further, and write UTM CRS info
    dem = dem.squeeze(drop=True).rename('elevation')
    try:
        proj_target_crs = proj.target_crs
    except AttributeError:  # No target_crs property for pyproj < 3.3.0!
        proj_target_crs = CRS(proj.to_proj4().split(' +step ')[-1])
    grid_crs = grid.copy().rio.write_crs(proj_target_crs)

    # Project DEM to UTM, further relabeling
    dem_utm = dem.rio.reproject_match(grid_crs, nodata=NODATA, resampling=Resampling.cubic_spline)
    dem_utm.rio.write_nodata(NODATA, inplace=True, encoded=True)
    units = dict(units='m')
    dem_utm.attrs = units
    warnings.warn('Elevation units are assumed to be in meters!', RTMWarning)
    for coordinate in 'x', 'y':
        dem_utm[coordinate].attrs = units

    # Create an output GeoTIFF if requested
    if output_file:

        # Create output raster filename/path
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        output_name = TEMPLATE.format(*grid.grid_center[::-1], grid.x_radius,
                                      grid.y_radius, grid.spacing)
        # Write to file
        output = os.path.join(OUTPUT_DIR, output_name)
        dem_utm.rio.to_raster(output, tags=dict(AREA_OR_POINT='Point'))

        print(f'Created output DEM file:\n\t{os.path.abspath(output)}')

    # Insert DEM data into the same form as the input grid
    dem_grid = grid.copy()
    dem_grid.data = dem_utm.data  # This will fail if the shapes are not identical!
    dem_grid.data[dem_grid.data == NODATA] = np.nan  # Set NODATA values to np.nan
    dem_grid.data[dem_grid.data < 0] = 0  # Set negative values (underwater?) to 0

    print('Done')

    if plot_output:

        print('Generating DEM hillshade plot...')

        projection = ccrs.UTM(**dem_grid.UTM)

        fig, ax = plt.subplots(figsize=(10, 10),
                               subplot_kw=dict(projection=projection))

        # Create hillshade
        shaded_dem = add_shading(dem_grid, azimuth=135, altitude=45)

        # Plot hillshade
        grid_shaded = grid.copy()
        grid_shaded.data = shaded_dem
        grid_shaded.plot.imshow(ax=ax, cmap='Greys_r', center=False,
                                add_colorbar=False, transform=projection)

        # Add translucent DEM
        im = dem_grid.plot.imshow(ax=ax, cmap='magma', alpha=0.5, vmin=0,
                             add_colorbar=False, transform=projection)
        cbar = fig.colorbar(im, label='Elevation (m)')
        cbar.solids.set_alpha(1)

        # Plot the center of the grid
        ax.scatter(*dem_grid.grid_center, s=50, color='limegreen',
                   edgecolor='black', label='Grid center',
                   transform=ccrs.PlateCarree())

        # Add a legend
        ax.legend(loc='best')

        if external_file:
            source_label = os.path.abspath(external_file)
        else:
            source_label = '1 arc-second SRTM data'

        ax.set_title('{}\nResampled to {} m spacing'.format(source_label,
                                                            dem_grid.spacing))

        fig.show()

        print('Done')

    return dem_grid


def grid_search(processed_st, grid, time_method, starttime=None, endtime=None,
                stack_method='sum', window=None, overlap=0.5, **time_kwargs):
    """
    Perform a grid search over :math:`x` and :math:`y` and return a 3-D object
    with dimensions :math:`(t, y, x)`. If a UTM grid is used, then the UTM
    :math:`(x, y)` coordinates for each station (`tr.stats.utm_x`,
    `tr.stats.utm_y`) are added to `processed_st`. Optionally provide a 2-D
    array of elevation values to enable 3-D distance computation.

    Args:
        processed_st (:class:`~obspy.core.stream.Stream`): Pre-processed data;
            output of :func:`~rtm.waveform.process_waveforms`
        grid (:class:`~xarray.DataArray`): Grid to use; output of
            :func:`define_grid`
        time_method (str): Method to use for calculating travel times. One of
            `'celerity'`, `'fdtd'`, or `'infresnel'`

                * `'celerity'` A single celerity is assumed for
                  propagation. Distances are either 2-D or 3-D (if a DEM
                  is supplied)

                * `'fdtd'` Travel times are calculated using a
                  finite-difference time-domain algorithm which accounts
                  for wave interactions with topography. Only valid for
                  UTM grids

                * `'infresnel'` Travel times are calculated using the shortest
                  diffracted path over topography and a single celerity

        starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time
            for grid search (default: `None`, which translates to
            `processed_st[0].stats.starttime`)
        endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for
            grid search (default: `None`, which translates to
            `processed_st[0].stats.endtime`)
        stack_method (str): Method to use for stacking aligned waveforms. One
            of `'sum'`, `'product'`, or `'semblance'` (default: `'sum'`)

                * `'sum'` Sum the aligned waveforms sample-by-sample

                * `'product'` Multiply the aligned waveforms
                  sample-by-sample. Results in a more spatially
                  concentrated stack maximum

                * `'semblance'` Multi-channel coherence computed over
                  defined time windows

        window (int or float): Time window [s] needed for `'semblance'`
            stacking (default: `None`)

        overlap (int or float): Overlap of time window for `'semblance'`
            stacking (default: `0.5`). Must be between 0 and 1, inclusive.

        **time_kwargs: Keyword arguments to be passed on to
            :func:`~rtm.travel_time.celerity_travel_time`,
            `~rtm.travel_time.fdtd_travel_time`, or
            :func:`~rtm.travel_time.infresnel_travel_time` functions. For details,
            see the docstrings of those functions.

    Returns:
        :class:`~xarray.DataArray` containing the 3-D :math:`(t, y, x)` stack
        function
    """

    # Check that the requested method works with the provided grid
    if time_method in ['fdtd', 'infresnel'] and not grid.UTM:
        raise NotImplementedError('The FDTD and infresnel methods are not implemented for '
                                  'unprojected (regional) grids.')

    # Get data dimensions
    npts_st = processed_st[0].stats.npts
    nsta = processed_st.count()

    # Use Stream times to define global time axis for S
    if stack_method == 'semblance':
        if not window:
            raise ValueError('Window must be defined for method '
                             f'\'{stack_method}\'.')

        times = np.arange(processed_st[0].stats.starttime,
                          processed_st[0].stats.endtime, window * (1 - overlap))

        # Define number of samples per window and increment
        winlen_samp = window * processed_st[0].stats.sampling_rate
        samp_inc = (1 - overlap) * winlen_samp

        # Sample pointer for window-based stack
        samples_stack = np.arange(0, npts_st, samp_inc)

        # Add final window to account for potential uneven number of samples
        if samples_stack[-1] < npts_st - 1:
            samples_stack = np.hstack((samples_stack, npts_st))
        samples_stack = samples_stack.astype(int, copy=False)

    else:
        # sample by sample-based stack
        times = processed_st[0].times(type='utcdatetime')

    # Expand grid dimensions in time
    S = grid.expand_dims(time=times.astype('datetime64[ns]')).copy()

    # Project stations in processed_st to UTM if necessary
    if grid.UTM:
        for tr in processed_st:
            tr.stats.utm_x, tr.stats.utm_y = _project_station_to_utm(tr, grid)
            tr.stats.utm_zone = grid.UTM['zone']

    # Call appropriate travel time array creation function
    if time_method == 'celerity':
        travel_times = celerity_travel_time(grid, processed_st, **time_kwargs)
        # Store celerity in S attributes
        S.attrs['celerity'] = time_kwargs['celerity']
    elif time_method == 'fdtd':
        travel_times = fdtd_travel_time(grid, processed_st, **time_kwargs)
    elif time_method == 'infresnel':
        travel_times = infresnel_travel_time(grid, processed_st, **time_kwargs)
        # Store celerity in S attributes
        S.attrs['celerity'] = time_kwargs['celerity']
    else:
        raise ValueError(f'Travel time calculation method \'{time_method}\' '
                         'not recognized. Method must be either \'celerity\', '
                         '\'fdtd\', or \'infresnel\'.')

    print('----------------------')
    print('PERFORMING GRID SEARCH')
    print(f'Method = \'{stack_method}\'')
    print('----------------------')

    st = processed_st.copy()

    # Determine the number of samples to be subtracted from travel times
    remove_samp = np.round(np.abs(travel_times.data) *
                           st[0].stats.sampling_rate).astype(int)
    remove_samp[remove_samp > npts_st] = 0

    # Create empty temporary data and stack arrays
    dtmp = np.zeros((nsta, npts_st))

    tic = time.time()

    bar = _grid_progress_bar(grid)
    for i, x in enumerate(S.x.values):
        for j, y in enumerate(S.y.values):
            for k, tr in enumerate(st):

                # Number of samples to subtract
                nrem = remove_samp[k, j, i]

                if nrem > 0:
                    # Number of zeroes for padding
                    nrem_zero = np.zeros(nrem)
                    dtmp[k, :] = np.hstack((tr.data[nrem:], nrem_zero))

                else:
                    dtmp[k, :] = tr.data

            if stack_method == 'sum':
                stk = np.sum(dtmp, axis=0)

            elif stack_method == 'product':
                stk = np.prod(dtmp, axis=0)

            elif stack_method == 'semblance':
                semb = []
                for t in range(len(samples_stack) - 1):
                    semb.append(calculate_semblance(
                        dtmp[:, samples_stack[t]:samples_stack[t + 1]]))
                stk = np.array(semb)

            else:
                raise ValueError(f'Stack method \'{stack_method}\' not '
                                 'recognized. Method must be either '
                                 '\'sum\' or \'product\'.')

            S.loc[dict(x=x, y=y)] = stk

            bar.update()

    # Remap for specified start and end times if provided
    if starttime:
        S = S[(S.time >= np.datetime64(starttime))]
    if endtime:
        S = S[(S.time <= np.datetime64(endtime))]

    # Mask by any NaN values in travel_times
    mask = np.broadcast_to(travel_times.isnull().all(axis=0).values, S.shape)
    S.data[mask] = np.nan

    toc = time.time()
    print(f'Done (elapsed time = {toc-tic:.1f} s)')

    if stack_method == 'product':
        warnings.warn('Watch out for zeros in the stack function due to zeroed '
                      'Traces with this stack method!', RTMWarning)

    return S


def calculate_time_buffer(grid, max_station_dist):
    """
    Utility function for estimating the amount of time needed for an infrasound
    signal to propagate from a source located anywhere in the RTM grid to the
    station farthest from the RTM grid center. This "travel time buffer" helps
    ensure that enough data is downloaded.

    Args:
        grid (:class:`~xarray.DataArray`): Grid to use; output of
            :func:`define_grid`
        max_station_dist (int or float): [km] The longest distance from the
            grid center to a station

    Returns:
        Maximum travel time [s] expected for a source anywhere in the grid to
        the station farthest from the grid center
    """

    # If projected grid, just calculate Euclidean distance for diagonal
    if grid.UTM:
        grid_diagonal = np.linalg.norm([grid.x_radius, grid.y_radius])  # [m]

    # If unprojected grid, find the "longer" of the two possible diagonals
    else:
        center_lon, center_lat = grid.grid_center
        corners = [(center_lat + grid.y_radius,
                    center_lon + grid.x_radius),
                   (center_lat - grid.y_radius,
                    center_lon - grid.x_radius)]
        diags = [gps2dist_azimuth(*corner, center_lat,
                                  center_lon)[0] for corner in corners]
        grid_diagonal = np.max(diags)  # [m]

    # Maximum distance a signal would have to travel is the longest distance
    # from the grid center to a station, PLUS the longest distance from the
    # grid center to a grid corner
    max_propagation_dist = max_station_dist * KM2M + grid_diagonal  # [m]

    # Calculate maximum travel time
    time_buffer = max_propagation_dist / MIN_CELERITY  # [s]

    return time_buffer


def _project_station_to_utm(tr, grid):
    """
    Projects `tr.latitude`, `tr.longitude` into the UTM zone of the input grid.
    Issues a warning if the coordinates of `tr` would locate to another UTM
    grid instead. (The implication here is that the user is trying to use an
    oversized UTM grid and is better off using an unprojected grid instead.)

    Args:
        tr: A :class:`~obspy.core.trace.Trace` containing station coordinates
        grid (:class:`~xarray.DataArray`): Projected :math:`(x, y)` grid; i.e.,
            output of :func:`define_grid` with `projected=True`

    Returns:
        List of [`utm_x`, `utm_y`] coordinates for station associated with `tr`
    """

    # Perform conversion to UTM
    proj = _proj_from_grid(grid)
    station_utm = proj.transform(tr.stats.latitude, tr.stats.longitude)

    # Check if station is outside of grid UTM zone
    station_zone_number = int(_estimate_utm_crs(tr.stats.latitude, tr.stats.longitude).utm_zone[:-1])
    grid_zone_number = grid.UTM['zone']

    if station_zone_number != grid_zone_number:
        warnings.warn(f'{tr.id} locates to UTM zone {station_zone_number} '
                      f'instead of grid UTM zone {grid_zone_number}. Consider '
                      'reducing station search extent or using an unprojected '
                      'grid.', RTMWarning)

    return station_utm
