import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
import matplotlib.patheffects as pe
from obspy.geodetics import gps2dist_azimuth
from .stack import get_peak_coordinates
import utm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from datetime import datetime

from . import RTMWarning


def plot_time_slice(S, processed_st, time_slice=None, label_stations=True,
                    hires=False, dem=None, plot_peak=True, cont_int=5,
                    annot_int=50):
    """
    Plot a time slice through :math:`S` to produce a map-view plot. If time is
    not specified, then the slice corresponds to the maximum of :math:`S` in
    the time direction. Can also plot the peak of the stack function over
    time.

    Args:
        S (:class:`~xarray.DataArray`): The stack function :math:`S`
        processed_st (:class:`~obspy.core.stream.Stream`): Pre-processed
            Stream; output of :func:`~rtm.waveform.process_waveforms` (This is
            needed because Trace metadata from this Stream are used to plot
            stations on the map)
        time_slice (:class:`~obspy.core.utcdatetime.UTCDateTime`): Time of
            desired time slice. The nearest time in :math:`S` to this specified
            time will be plotted. If `None`, the time corresponding to
            :math:`\max(S)` is used (default: `None`)
        label_stations (bool): Toggle labeling stations with network and
            station codes (default: `True`)
        hires (bool): If `True`, use higher-resolution background
            image/coastlines, which looks better but can be slow (default:
            `False`)
        dem (:class:`~xarray.DataArray`): Overlay time slice on a user-supplied
            DEM from :class:`~rtm.grid.produce_dem` (default: `None`)
        plot_peak (bool): Plot the peak stack function over time as a subplot
            (default: `True`)
        cont_int (int): Contour interval [m] for plots with DEM data
        annot_int (int): Annotated contour interval [m] for plots with DEM data
            (these contours are thicker and labeled)

    Returns:
        :class:`~matplotlib.figure.Figure`: Output figure
    """

    # Don't plot peak of stack function when length of stack is one
    if plot_peak and len(S.time) == 1:
        plot_peak = False
        warnings.warn('Stack time length = 1, not plotting peak', RTMWarning)

    st = processed_st.copy()

    # Get coordinates of stack maximum in (latitude, longitude)
    time_max, y_max, x_max, peaks, props = get_peak_coordinates(S, unproject=S.UTM)

    # Gather coordinates of grid center
    lon_0, lat_0 = S.grid_center

    if dem is not None:

        # Note that the below is a hacky way to use matplotlib instead of
        # cartopy and should be edited once cartopy labeling is functional
        proj = None
        transform = None
        plot_transform = None
        lon_0, lat_0, _, _ = utm.from_latlon(S.grid_center[1], S.grid_center[0])
        x_max, y_max, _, _ = utm.from_latlon(y_max, x_max)
        for tr in st:
            tr.stats.longitude, tr.stats.latitude, _, _ = utm.from_latlon(
                tr.stats.latitude, tr.stats.longitude)

    elif S.UTM:
        proj = ccrs.UTM(**S.UTM)
        transform = proj
        plot_transform = ccrs.PlateCarree()
    else:
        # This is a good projection to use since it preserves area
        proj = ccrs.AlbersEqualArea(central_longitude=lon_0,
                                    central_latitude=lat_0,
                                    standard_parallels=(S.y.values.min(),
                                                        S.y.values.max()))
        transform = ccrs.PlateCarree()
        plot_transform = ccrs.PlateCarree()

    if plot_peak:
        fig, (ax, ax1) = plt.subplots(figsize=(8, 12), nrows=2,
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      subplot_kw=dict(projection=proj))

        #axes kluge so the second one can have a different projection
        ax1.remove()
        ax1 = fig.add_subplot(414)

    else:
        fig, ax = plt.subplots(figsize=(8, 8),
                               subplot_kw=dict(projection=proj))

    # In either case, we convert from UTCDateTime to np.datetime64
    if time_slice:
        time_to_plot = np.datetime64(time_slice)
    else:
        time_to_plot = np.datetime64(time_max)

    slice = S.sel(time=time_to_plot, method='nearest')

    #convert UTM grid/etc to x/y coordinates with (0,0) as origin
    if S.UTM and dem is not None:

        print('Converting to x/y grid')

        #update dataarrays to x/y coordinates from dem
        xmin = dem.x.data.min() + dem.x_radius
        ymin = dem.y.data.min() + dem.y_radius
        slice = slice.assign_coords(x=(slice.x.data - xmin))
        slice = slice.assign_coords(y=(slice.y.data - ymin))
        dem = dem.assign_coords(x=(dem.x.data - xmin))
        dem = dem.assign_coords(y=(dem.y.data - ymin))
        lon_0 = lon_0 - xmin
        lat_0 = lat_0 - ymin
        x_max = x_max - xmin
        y_max = y_max - ymin
        for tr in st:
            tr.stats.longitude = tr.stats.longitude - xmin
            tr.stats.latitude = tr.stats.latitude - ymin

    if dem is None:
        _plot_geographic_context(ax=ax, utm=S.UTM, hires=hires)
        slice_plot_kwargs = dict(ax=ax, alpha=0.5, cmap='viridis',
                                 add_colorbar=False, transform=transform)
    else:
        # Rounding to nearest cont_int
        all_levels = np.arange(np.ceil(dem.min().data / cont_int),
                               np.floor(dem.max().data / cont_int) + 1) * cont_int
        # Rounding to nearest annot_int
        annot_levels = np.arange(np.ceil(dem.min().data / annot_int),
                                 np.floor(dem.max().data / annot_int) + 1) * annot_int
        # Ensure we don't draw annotated levels twice
        cont_levels = []
        for level in all_levels:
            if level not in annot_levels:
                cont_levels.append(level)

        dem.plot.contour(ax=ax, colors='k', levels=cont_levels, zorder=-1,
                         linewidths=0.3)
        # Use thicker lines for annotated contours
        cs = dem.plot.contour(ax=ax, colors='k', levels=annot_levels,
                              zorder=-1, linewidths=0.7)
        ax.clabel(cs, fontsize=9, fmt='%d', inline=True)  # Actually annotate

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        slice_plot_kwargs = dict(ax=ax, alpha=0.7, cmap='viridis',
                                 add_colorbar=False, add_labels=False,
                                 zorder=0)

        plot_transform = ax.transData

        # Mask areas outside of DEM extent
        dem_slice = dem.sel(x=slice.x, y=slice.y, method='nearest')  # Select subset of DEM that slice occupies
        slice.data[np.isnan(dem_slice.data)] = np.nan

    if S.UTM:
        # imshow works well here (no gridlines in translucent plot)
        sm = slice.plot.imshow(**slice_plot_kwargs)
    else:
        # imshow performs poorly for Albers equal-area projection - use
        # pcolormesh instead (gridlines will show in translucent plot)
        sm = slice.plot.pcolormesh(**slice_plot_kwargs)

    # Initialize list of handles for legend
    h = [None, None, None]
    scatter_zorder = 5

    # Plot center of grid
    h[0] = ax.scatter(lon_0, lat_0, s=50, color='limegreen', edgecolor='black',
                      label='Grid center', transform=plot_transform,
                      zorder=scatter_zorder)

    # Plot stack maximum
    if S.UTM:
        # x/y formatting
        label = 'Stack max'
    else:
        # Lat/lon formatting
        label = f'Stack max\n({y_max:.4f}, {x_max:.4f})'
    h[1] = ax.scatter(x_max, y_max, s=100, color='red', marker='*',
                      edgecolor='black', label=label,
                      transform=plot_transform, zorder=scatter_zorder)

    # Plot stations
    for tr in st:
        h[2] = ax.scatter(tr.stats.longitude, tr.stats.latitude, marker='v',
                          color='orange', edgecolor='black',
                          label='Station', transform=plot_transform,
                          zorder=scatter_zorder)
        if label_stations:
            ax.text(tr.stats.longitude, tr.stats.latitude,
                    '  {}.{}'.format(tr.stats.network, tr.stats.station),
                    verticalalignment='center_baseline',
                    horizontalalignment='left', fontsize=10, color='white',
                    transform=plot_transform, zorder=scatter_zorder,
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'),
                                  pe.Normal()],
                    clip_on=True)

    ax.legend(h, [handle.get_label() for handle in h], loc='best',
              framealpha=1, borderpad=.3, handletextpad=.3)

    time_round = np.datetime64(slice.time.values + np.timedelta64(500, 'ms'), 's').astype(datetime)  # Nearest second
    title = 'Time: {}'.format(time_round)

    if hasattr(S, 'celerity'):
        title += f'\nCelerity: {S.celerity:g} m/s'

    # Label global maximum if applicable
    if slice.time.values == time_max:
        title = 'GLOBAL MAXIMUM\n\n' + title

    ax.set_title(title, pad=20)

    # Another hack that can be removed once cartopy is improved
    if dem is not None:
        ax.set_aspect('equal')

    # crop plot to show just the slice area
    ax.set_xlim(slice.x.min(), slice.x.max())
    ax.set_ylim(slice.y.min(), slice.y.max())

    ax_pos = ax.get_position()
    cloc = [ax_pos.x1+.02, ax_pos.y0, .02, ax_pos.height]
    cbaxes = fig.add_axes(cloc)
    cbar = fig.colorbar(sm, cax=cbaxes, label='Stack amplitude')
    cbar.solids.set_alpha(1)

    if plot_peak:
        plot_stack_peak(S, plot_max=True, ax=ax1)

    fig.show()

    return fig


def plot_record_section(st, origin_time, source_location, plot_celerity=None,
                        label_waveforms=True):
    """
    Plot a record section based upon user-provided source location and origin
    time. Optionally plot celerity for reference, with two plotting options.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Any Stream object with
            `tr.stats.latitude`, `tr.stats.longitude` attached
        origin_time (:class:`~obspy.core.utcdatetime.UTCDateTime`): Origin time
            for record section
        source_location (tuple): Tuple of (`lat`, `lon`) specifying source
            location
        plot_celerity: Can be either `'range'` or a single celerity or a list
            of celerities. If `'range'`, plots a continuous swatch of
            celerities from 260-380 m/s. Otherwise, plots specific celerities.
            If `None`, does not plot any celerities (default: `None`)
        label_waveforms (bool): Toggle labeling waveforms with network and
            station codes (default: `True`)

    Returns:
        :class:`~matplotlib.figure.Figure`: Output figure
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

        # Otherwise, they provided specific celerities
        else:
            # Type conversion
            if type(plot_celerity) is not list:
                plot_celerity = [plot_celerity]

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

    time_round = np.datetime64(origin_time + 0.5, 's').astype(datetime)  # Nearest second
    ax.set_xlabel('Time (s) from {}'.format(time_round))
    ax.set_ylabel('Distance (km) from '
                  '({:.4f}, {:.4f})'.format(*source_location))

    fig.tight_layout()
    fig.show()

    return fig


def plot_st(st, filt, equal_scale=False, remove_response=False,
            label_waveforms=True):
    """
    Plot Stream waveforms in a publication-quality figure. Multiple plotting
    options, including filtering.

    Args:
        st (:class:`~obspy.core.stream.Stream`): Any Stream object
        filt (list): A two-element list of lower and upper corner frequencies
            for filtering. Specify `None` if no filtering is desired.
        equal_scale (bool): Set equal scale for all waveforms (default:
            `False`)
        remove_response (bool): Remove response by applying sensitivity
        label_waveforms (bool): Toggle labeling waveforms with network and
            station codes (default: `True`)

    Returns:
        :class:`~matplotlib.figure.Figure`: Output figure
    """

    st_plot = st.copy()
    ntra = len(st)
    tvec = st_plot[0].times('matplotlib')

    if remove_response:
        print('Applying sensitivity')
        st_plot.remove_sensitivity()

    if filt:
        print('Filtering between %.1f-%.1f Hz' % (filt[0], filt[1]))

        st_plot.detrend(type='linear')
        st_plot.taper(max_percentage=.01)
        st_plot.filter("bandpass", freqmin=filt[0], freqmax=filt[1], corners=2,
                       zerophase=True)

    if equal_scale:
        ym = np.max(st_plot.max())

    fig, ax = plt.subplots(figsize=(8, 6), nrows=ntra, sharex=True)

    for i, tr in enumerate(st_plot):
        ax[i].plot(tvec, tr.data, 'k-')
        ax[i].set_xlim(tvec[0], tvec[-1])
        if equal_scale:
            ax[i].set_ylim(-ym, ym)
        else:
            ax[i].set_ylim(-tr.data.max(), tr.data.max())
        plt.locator_params(axis='y', nbins=4)
        ax[i].tick_params(axis='y', labelsize=8)
        ax[i].ticklabel_format(useOffset=False, style='plain')

        if tr.stats.channel[1] == 'D':
            ax[i].set_ylabel('Pressure [Pa]', fontsize=8)
        else:
            ax[i].set_ylabel('Velocity [m/s]', fontsize=8)

        if label_waveforms:
            ax[i].text(.85, .9,
                       f'{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}',
                       verticalalignment='center', transform=ax[i].transAxes)

    # Tick locating and formatting
    locator = mdates.AutoDateLocator()
    ax[-1].xaxis.set_major_locator(locator)
    ax[-1].xaxis.set_major_formatter(_UTCDateFormatter(locator))
    fig.autofmt_xdate()

    fig.tight_layout()
    plt.subplots_adjust(hspace=.12)
    fig.show()

    return fig


def plot_stack_peak(S, plot_max=False, ax=None):
    """
    Plot the stack function (at the spatial stack max) as a function of time.

    Args:
        S: :class:`~xarray.DataArray` containing the stack function :math:`S`
        plot_max (bool): Plot maximum value with red circle (default: `False`)
        ax (:class:`~matplotlib.axes.Axes`): Pre-existing axes to plot into

    Returns:
        :class:`~matplotlib.figure.Figure`: Output figure
    """

    s_peak = S.max(axis=(1, 2)).data

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()  # Get figure to which provided axis belongs
    ax.plot(S.time, s_peak, 'k-')
    if plot_max:
        stack_maximum = S.where(S == S.max(), drop=True).squeeze()
        marker_kwargs = dict(marker='*', color='red', edgecolor='black', s=150,
                             zorder=5, clip_on=False)
        if stack_maximum.size > 1:
            max_indices = np.argwhere(~np.isnan(stack_maximum.data))
            ax.scatter(stack_maximum[tuple(max_indices[0])].time.data,
                       stack_maximum[tuple(max_indices[0])].data,
                       **marker_kwargs)
            warnings.warn(f'Multiple global maxima ({len(stack_maximum.data)}) '
                          'present in S!', RTMWarning)
        else:
            ax.scatter(stack_maximum.time.data, stack_maximum.data,
                       **marker_kwargs)

    ax.set_xlim(S.time[0].data, S.time[-1].data)
    ax.set_ylim(bottom=0)  # Never can go below zero
    ax.set_ylabel('Max stack amplitude')

    # Tick locating and formatting
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(_UTCDateFormatter(locator))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    return fig


def _plot_geographic_context(ax, utm, hires=False):
    """
    Plot geographic basemap information on a map axis. Plots a background image
    for UTM-projected plots and simple coastlines for unprojected plots.

    Args:
        ax (:class:`~cartopy.mpl.geoaxes.GeoAxes`): Existing axis to plot into
        utm (bool): Flag specifying if the axis is projected to UTM or not
        hires (bool): If `True`, use higher-resolution images/coastlines
            (default: `False`)
    """

    # Since projected grids cover less area and may not include coastlines,
    # use a background image to provide geographical context (can be slow)
    if utm:
        if hires:
            zoom_level = 12
        else:
            zoom_level = 8
        ax.add_image(Stamen(style='terrain-background'), zoom_level, zorder=-1)

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


# Subclass ConciseDateFormatter (modifies __init__() and set_axis() methods)
class _UTCDateFormatter(mdates.ConciseDateFormatter):
    def __init__(self, locator, tz=None):
        super().__init__(locator, tz=tz, show_offset=True)

        # Re-format datetimes
        self.formats[5] = '%H:%M:%S.%f'
        self.zero_formats = self.formats
        self.offset_formats = [
            'UTC time',
            'UTC time in %Y',
            'UTC time in %B %Y',
            'UTC time on %Y-%m-%d',
            'UTC time on %Y-%m-%d',
            'UTC time on %Y-%m-%d',
        ]

    def set_axis(self, axis):
        self.axis = axis

        # If this is an x-axis (usually is!) then center the offset text
        if self.axis.axis_name == 'x':
            offset = self.axis.get_offset_text()
            offset.set_horizontalalignment('center')
            offset.set_x(0.5)
