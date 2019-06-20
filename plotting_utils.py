import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
from obspy.geodetics import gps2dist_azimuth
import numpy as np
import warnings


plt.ioff()  # Don't show the figure unless fig.show() is explicitly called

FONT_SIZE = 14
plt.rcParams.update({'font.size': FONT_SIZE})


def plot_time_slice(S, processed_st, time_slice=None, celerity_slice=None):
    """
    Plot a time slice through S to produce a map-view plot. If time and
    celerity are not specified, then the slice corresponds to the maximum of S
    in the time and celerity directions.

    Args:
        S: xarray.DataArray containing the stack function S
        processed_st: Pre-processed Stream <-- output of process_waveforms()
                      (This is need because Trace metadata from this Stream are
                      used to plot stations on the map)
        time_slice: UTCDateTime of desired time slice. The nearest time in S to
                    this specified time will be plotted. If None, the time
                    corresponding to max(S) is used (default: None)
        celerity_slice: [m/s] Value of celerity to use for slice. Throws an
                        error if the specified celerity is not in S. If None,
                        the celerity corresponding to max(S) is used (default:
                        None)
    Returns:
        fig: Output figure
    """

    # Get coordinates of stack maximum
    stack_maximum = S.where(S == S.max(), drop=True)
    if stack_maximum.shape[0] is not 1:
        warnings.warn('Multiple maxima present in S along the time dimension. '
                      'Using first occurrence.')
    elif stack_maximum.shape[1] is not 1:
        warnings.warn('Multiple maxima present in S along the celerity '
                      'dimension. Using first occurrence.')
    max_coords = stack_maximum[0, 0, 0, 0].coords
    time_max = max_coords['time'].values
    celerity_max = max_coords['celerity'].values

    # Gather coordinates of grid center
    lon_0, lat_0 = S.attrs['grid_center']

    if S.attrs['UTM']:
        proj = ccrs.UTM(**S.attrs['UTM'])
        transform = proj
    else:
        # This is a good projection to use since it preserves area
        proj = ccrs.AlbersEqualArea(central_longitude=lon_0,
                                    central_latitude=lat_0,
                                    standard_parallels=(S['y'].values.min(),
                                                        S['y'].values.max()))
        transform = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw=dict(projection=proj))

    _plot_geographic_context(ax=ax, utm=S.attrs['UTM'])

    if time_slice:
        time_to_plot = np.datetime64(time_slice)
    else:
        time_to_plot = time_max

    if celerity_slice:
        if celerity_slice not in S['celerity'].values:
            raise KeyError(f'Celerity {celerity_slice} m/s is not in S.')
        celerity_to_plot = celerity_slice
    else:
        celerity_to_plot = celerity_max

    slice = S.sel(time=time_to_plot, celerity=celerity_to_plot,
                  method='nearest')

    qm = slice.plot.pcolormesh(ax=ax, alpha=0.5, cmap='inferno',
                               add_colorbar=False, transform=transform)

    cbar = fig.colorbar(qm, label='Stack amplitude', pad=0.1)
    cbar.solids.set_alpha(1)

    # Plot center of grid
    ax.scatter(lon_0, lat_0, s=100, color='red', marker='*',
               transform=ccrs.Geodetic())

    # Plot stations
    for tr in processed_st:
        ax.scatter(tr.stats.longitude,  tr.stats.latitude, color='black',
                   transform=ccrs.Geodetic())
        ax.text(tr.stats.longitude, tr.stats.latitude,
                '  {}.{}'.format(tr.stats.network, tr.stats.station),
                verticalalignment='center_baseline',
                horizontalalignment='left',
                transform=ccrs.Geodetic())

    title = f'Time: {slice.time.values}\nCelerity: {slice.celerity.values} m/s'

    # Label global maximum if applicable
    if slice.time.values == time_max and slice.celerity.values == celerity_max:
        title = 'GLOBAL MAXIMUM\n\n' + title

    ax.set_title(title, pad=20)

    fig.canvas.draw()
    fig.tight_layout()
    fig.show()

    return fig


def plot_record_section(st, origin_time, source_location, plot_celerity=None):
    """
    Plot a record section based upon user-provided source location and origin
    time. Optionally plot celerity for reference, with two plotting options.

    Args:
        st: Any Stream object with tr.stats.latitude, tr.stats.longitude data
        origin_time: UTCDateTime specifying the origin time
        source_location: Tuple of (lat, lon) specifying source location
        plot_celerity: Can be either 'range' or a list of celerities. If
                       'range', plots a continuous swatch of celerities from
                       260-380 m/s. If a list, plots these specific celerities.
                       If None, does not plot any celerities (default: None)
    Returns:
        fig: Output figure
    """

    st_edit = st.copy()

    for tr in st_edit:
        tr.stats.distance, _, _ = gps2dist_azimuth(*source_location,
                                                   tr.stats.latitude,
                                                   tr.stats.longitude)

    st_edit.trim(origin_time)

    fig = plt.figure(figsize=(12, 8))

    st_edit.plot(fig=fig, type='section', orientation='horizontal',
                 fillcolors=('black', 'black'))

    ax = fig.axes[0]

    if plot_celerity:

        # Check if user requested a continuous range of celerities
        if plot_celerity == 'range':
            celerity_list = np.arange(260, 380 + 1, 1)
            zorder = -1

        # Otherwise, they provided a list of discrete celerities
        else:
            celerity_list = plot_celerity
            celerity_list.sort()
            zorder = None

        # Create colormap of appropriate length
        cmap = plt.cm.get_cmap('rainbow', len(celerity_list))
        colors = [cmap(i) for i in range(cmap.N)]

        xlim = np.array(ax.get_xlim())

        for celerity, color in zip(celerity_list, colors):
            ax.plot(xlim, xlim * celerity / 1000, label=celerity, color=color,
                    zorder=zorder)

        # If plotting a continuous range, add a colorbar
        if plot_celerity == 'range':
            mapper = plt.cm.ScalarMappable(cmap=cmap)
            mapper.set_array(celerity_list)
            fig.colorbar(mapper, label='Celerity (m/s)')

        # If plotting discrete celerities, just add a legend
        else:
            ax.legend(title='Celerity (m/s)', loc='lower right')

    ax.set_ylim(bottom=0)  # Show all the way to zero offset

    ax.set_xlabel(f'Time (s) from {origin_time.datetime}')
    ax.set_ylabel('Distance (km) from '
                  '({:.4f}, {:.4f})'.format(*source_location))

    fig.canvas.draw()
    fig.tight_layout()
    fig.show()

    return fig


def _plot_geographic_context(ax, utm):
    """
    Plot geographic basemap information on a map axis. Plots a background image
    for UTM-projected plots and simple coastlines for unprojected plots.

    Args:
        ax: Existing GeoAxes to plot into
        utm: Flag specifying if the axis is projected to UTM or not
    """

    # Since projected grids cover less area and may not include coastlines,
    # use a background image to provide geographical context (can be slow)
    if utm:
        zoom_level = 8
        ax.add_image(Stamen(style='terrain-background'), zoom_level)

    # Since unprojected grids have regional/global extent, just show the
    # coastlines
    else:
        scale = '50m'
        land = cfeature.LAND.with_scale(scale)
        ax.add_feature(land, facecolor=cfeature.COLORS['land'],
                       edgecolor='black')
        ax.background_patch.set_facecolor(cfeature.COLORS['water'])
        lakes = cfeature.LAKES.with_scale(scale)
        ax.add_feature(lakes, facecolor=cfeature.COLORS['water'],
                       edgecolor='black', zorder=0)
