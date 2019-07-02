import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
import numpy as np
import warnings
from . import RTMWarning


def plot_time_slice(S, processed_st, time_slice=None, celerity_slice=None,
                    label_stations=True, hires=False):
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
        label_stations: Toggle labeling stations with network and station codes
                        (default: True)
        hires: If True, use higher-resolution background image/coastlines,
               which looks better but can be slow (default: False)
    Returns:
        fig: Output figure
    """

    # Get coordinates of stack maximum
    stack_maximum = S.where(S == S.max(), drop=True)
    if stack_maximum.shape[0] is not 1:
        warnings.warn('Multiple maxima present in S along the time dimension. '
                      'Using first occurrence.', RTMWarning)
    if stack_maximum.shape[1] is not 1:
        warnings.warn('Multiple maxima present in S along the celerity '
                      'dimension. Using first occurrence.', RTMWarning)
    max_coords = stack_maximum[0, 0, 0, 0].coords
    time_max = max_coords['time'].values
    celerity_max = max_coords['celerity'].values
    x_max = max_coords['x'].values
    y_max = max_coords['y'].values

    # Gather coordinates of grid center
    lon_0, lat_0 = S.attrs['grid_center']

    if S.attrs['UTM']:
        proj = ccrs.UTM(**S.attrs['UTM'])
        transform = proj
        # For UTM, label as (x, y)
        max_label = '({:.1f} m E, {:.1f} m N)'.format(x_max, y_max)
    else:
        # This is a good projection to use since it preserves area
        proj = ccrs.AlbersEqualArea(central_longitude=lon_0,
                                    central_latitude=lat_0,
                                    standard_parallels=(S['y'].values.min(),
                                                        S['y'].values.max()))
        transform = ccrs.PlateCarree()
        # For lat/lon, label as (lat, lon)
        max_label = '({:.4f}, {:.4f})'.format(y_max, x_max)

    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw=dict(projection=proj))

    _plot_geographic_context(ax=ax, utm=S.attrs['UTM'], hires=hires)

    if time_slice:
        time_to_plot = np.datetime64(time_slice)
    else:
        time_to_plot = time_max

    if celerity_slice:
        if celerity_slice not in S['celerity'].values:
            raise IndexError(f'Celerity {celerity_slice} m/s is not in S.')
        celerity_to_plot = celerity_slice
    else:
        celerity_to_plot = celerity_max

    slice = S.sel(time=time_to_plot, celerity=celerity_to_plot,
                  method='nearest')

    qm = slice.plot.pcolormesh(ax=ax, alpha=0.5, cmap='inferno',
                               add_colorbar=False, transform=transform)

    cbar = fig.colorbar(qm, label='Stack amplitude', aspect=30)
    cbar.solids.set_alpha(1)

    # Initialize list of handles for legend
    h = [None, None, None]

    # Plot center of grid
    h[0] = ax.scatter(lon_0, lat_0, s=50, color='limegreen', edgecolor='black',
                      label='Grid center', transform=ccrs.Geodetic())

    # Plot stack maximum
    h[1] = ax.scatter(x_max, y_max, s=100, color='red', marker='*',
                      edgecolor='black', label='Stack maximum\n' + max_label,
                      transform=transform)

    # Plot stations
    for tr in processed_st:
        h[2] = ax.scatter(tr.stats.longitude,  tr.stats.latitude, marker='v',
                          color='white', edgecolor='black',
                          label='Infrasound sensor', transform=ccrs.Geodetic())
        if label_stations:
            ax.text(tr.stats.longitude, tr.stats.latitude,
                    '  {}.{}'.format(tr.stats.network, tr.stats.station),
                    verticalalignment='center_baseline',
                    horizontalalignment='left', fontsize=10,
                    transform=ccrs.Geodetic())

    ax.legend(h, [handle.get_label() for handle in h], loc='best')

    title = f'Time: {UTCDateTime(str(slice.time.values)).datetime}' \
            f'\nCelerity: {slice.celerity.values:g} m/s'

    # Label global maximum if applicable
    if slice.time.values == time_max and slice.celerity.values == celerity_max:
        title = 'GLOBAL MAXIMUM\n\n' + title

    ax.set_title(title, pad=20)

    fig.canvas.draw()  # Needed to make fig.tight_layout() work
    fig.tight_layout()
    fig.show()

    return fig


def plot_record_section(st, origin_time, source_location, plot_celerity=None,
                        label_waveforms=True):
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
        label_waveforms: Toggle labeling waveforms with network and station
                         codes (default: True)
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

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    if label_waveforms:
        for tr in st_edit:
            ax.text(1.01, tr.stats.distance / 1000,
                    f'{tr.stats.network}.{tr.stats.station}',
                    verticalalignment='center', transform=trans, fontsize=10)
        pad = 0.1  # Move colorbar to the right to make room for labels
    else:
        pad = 0.05  # Matplotlib default for vertical colorbars

    if plot_celerity:

        # Check if user requested a continuous range of celerities
        if plot_celerity == 'range':
            inc = 0.5  # [m/s]
            celerity_list = np.arange(220, 350 + inc, inc)  # [m/s] Includes
                                                            # all reasonable
                                                            # celerities
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
        y_max = ax.get_ylim()[1]  # Save this for re-scaling axis

        for celerity, color in zip(celerity_list, colors):
            ax.plot(xlim, xlim * celerity / 1000, label=f'{celerity:g}',
                    color=color, zorder=zorder)

        ax.set_ylim(top=y_max)  # Scale y-axis to pre-plotting extent

        # If plotting a continuous range, add a colorbar
        if plot_celerity == 'range':
            mapper = plt.cm.ScalarMappable(cmap=cmap)
            mapper.set_array(celerity_list)
            cbar = fig.colorbar(mapper, label='Celerity (m/s)', pad=pad,
                                aspect=30)
            cbar.ax.minorticks_on()

        # If plotting discrete celerities, just add a legend
        else:
            ax.legend(title='Celerity (m/s)', loc='lower right', framealpha=1,
                      edgecolor='inherit')

    ax.set_ylim(bottom=0)  # Show all the way to zero offset

    ax.set_xlabel(f'Time (s) from {origin_time.datetime}')
    ax.set_ylabel('Distance (km) from '
                  '({:.4f}, {:.4f})'.format(*source_location))

    fig.tight_layout()
    fig.show()

    return fig


def _plot_geographic_context(ax, utm, hires=False):
    """
    Plot geographic basemap information on a map axis. Plots a background image
    for UTM-projected plots and simple coastlines for unprojected plots.

    Args:
        ax: Existing GeoAxes to plot into
        utm: Flag specifying if the axis is projected to UTM or not
        hires: If True, use higher-resolution images/coastlines (default:
               False)
    """

    # Since projected grids cover less area and may not include coastlines,
    # use a background image to provide geographical context (can be slow)
    if utm:
        if hires:
            zoom_level = 12
        else:
            zoom_level = 8
        ax.add_image(Stamen(style='terrain-background'), zoom_level)

    # Since unprojected grids have regional/global extent, just show the
    # coastlines
    else:
        if hires:
            scale = '10m'
        else:
            scale = '50m'
        land = cfeature.LAND.with_scale(scale)
        ax.add_feature(land, facecolor=cfeature.COLORS['land'],
                       edgecolor='black')
        ax.background_patch.set_facecolor(cfeature.COLORS['water'])
        lakes = cfeature.LAKES.with_scale(scale)
        ax.add_feature(lakes, facecolor=cfeature.COLORS['water'],
                       edgecolor='black', zorder=0)
