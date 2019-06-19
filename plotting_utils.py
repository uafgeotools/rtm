import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import warnings


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

    scale = '50m'
    land = cfeature.LAND.with_scale(scale)
    ax.add_feature(land, facecolor=cfeature.COLORS['land'],
                   edgecolor='black')
    ax.background_patch.set_facecolor(cfeature.COLORS['water'])
    lakes = cfeature.LAKES.with_scale(scale)
    ax.add_feature(lakes, facecolor=cfeature.COLORS['water'],
                   edgecolor='black', zorder=0)

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

    # If time slice corresponds to max(S), label as such
    if slice['time'].values == time_max:
        is_max_time = r' $\bf[MAX]$'
    else:
        is_max_time = ''

    # If celerity slice corresponds to max(S), label as such
    if slice['celerity'].values == celerity_max:
        is_max_celerity = r' $\bf[MAX]$'
    else:
        is_max_celerity = ''

    title = f'Time: {slice.time.values}{is_max_time}\n' \
            f'Celerity: {slice.celerity.values} m/s{is_max_celerity}'
    ax.set_title(title, pad=20)

    return fig
