import os
import json
import utm
import numpy as np
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth
import re
import glob


def prepare_fdtd_run(FDTD_DIR, FILENAME_ROOT, station, dem, H_MAX, TEMP, MAX_T,
                     DT, SRC_FREQ, VSNAP, SURSNAP, SNAPOUT):
    """
    Prepare and write RTM/FDTD files. Writes station, elevation, density, and
    sound speed file. Also parameter files for each station and shell script
    for FDTD calculation

    Args:
        FDTD_DIR: output directory for FDTD run
        FILENAME_ROOT: FDTD input filename prefix
        station: SEED station code
        dem: xarray.DataArray object containing the elevation values as well as
             grid coordinates and metadata
        H_MAX: max grid height [m]
        TEMP: temerature for sound speed calculation [K]
        MAX_T: duration of FDTD simulation [s] (make sure it extends across
               your grid)
        DT: simulation time (dt <= dh/c*np.sqrt(3))
        SRC_FREQ: source frequency [Hz] (make sure at least 20 wavelength per
                  dh)
        VSNAP: vertical output snapshot
        SURSNAP: surface pressure output snapshot
        SNAPPOUT: snapshot output interval [s]
    """

    print('--------------')
    print('CREATING RTM INPUT FILES FOR FDTD')
    print('--------------')

    plotcheck=0 #plot stations on DEM as a check

    r = 287.058    # [J/mol*K]; universal gas constant
    rho = 101325/(r*TEMP)    # air density calculation
    c = np.sqrt(1.402 * r * TEMP) # [m/s]; adiabatic sound speed

    #set up x/y grid and DEM
    x=np.array(dem.x-dem.x.min())
    y=np.array(dem.y-dem.y.min())
    xmax=x.max()
    ymax=y.max()

    print ('Max_x = ' + str(xmax))
    print('Max_y = ' + str(ymax))
    print('Max Z = '  + str(H_MAX))
    print('Min H = 0')
    print ('dh = ' + str(dem.spacing))

    # Save DEM into one-column text file from lower-left to upper-right, row by
    # row
    if not os.path.isdir(FDTD_DIR + 'input/'):
        os.makedirs(FDTD_DIR + 'input/')
    topo_file = FDTD_DIR + 'input/' + 'elev_' + FILENAME_ROOT + '.txt'

    #unravel elevation to write to a file
    elev=np.ravel(dem)
    elev[elev<0]=0

    #now deal with stations
    station_file = FDTD_DIR + 'input/' + 'sta_' + FILENAME_ROOT + '.txt'

    with open('watc_infra_coords.json') as f:
        WATC_INFRA_COORDS = json.load(f)

    #get station lat/lon and utm coordinates
    staloc={}   #lat,lon,z
    stautm={}   #utmx,utmy,utmzone
    staxyz={}   #x,y,z for FDTD grid
    for i,sta in enumerate(station):
        try:
            staloc[i] = WATC_INFRA_COORDS[sta]
            stautm[i] = utm.from_latlon(staloc[i][0], staloc[i][1])
            #find station x/y grid point closest to utm x/y
            staxyz[i] = [np.abs(dem.x.values-stautm[i][0]).argmin(),
                  np.abs(dem.y.values-stautm[i][1]).argmin(),staloc[i][2]]
        except KeyError:
           print('Failed! No matching station coordinates found for %s'%sta)
           raise

    #plot stations/etc on DEM as a check
    if plotcheck:
        line_s = np.arange(0,H_MAX,20)

        fig1=plt.figure(1)
        fig1.set_size_inches(4.5,6)
        plt.clf()
        ax=plt.subplot(111)
        #ax2.contour(x,y,z,ls,colors='k',linewidths=.35)
        ax.imshow(dem,origin='lower', extent=[min(x), max(x), min(y), max(y)],
                  cmap='jet')
        ax.contour(x,y,dem,line_s,colors='k',linewidths=.35)
        ax.set_aspect('equal')
        for i,sta in enumerate(station):
            ax.plot(x[staxyz[i][0]],y[staxyz[i][1]],'bo')
            ax.text(x[staxyz[i][0]]+10,y[staxyz[i][1]]+10,sta)

    # save files for FDTD input
    print('Saving elevation file...%d values'%len(elev))
    f = open(topo_file,'w')
    #elevation header: x,y,dx,dy
    f.write(str(len(x)) + ' ' + str(len(y)) + ' ' + str(float(dem.spacing)) +
            ' ' + str(float(dem.spacing)) +'\n')
    for ii in range(len(elev)):
        #if np.remainder(temp_elev, grid.spacing) == 1:
        #    temp_elev = temp_elev + 1
        f.write(str(int(round(elev[ii]))) +'\n')
    f.close()
    print('Done')

    print('Saving station file')
    f = open(station_file,'w')
    for i,sta in enumerate(station):
        temp_x = int(round(staxyz[i][0]))
        temp_y = int(round(staxyz[i][1]))
        temp_z = int(round(staxyz[i][2]))
        f.write(sta + ' ' + str(temp_x) + ' ' + str(temp_y) + ' ' + str(temp_z)
                + '\n')
    f.close()

    # create vertical profiles for FDTD. Static values for now
    num_rows = int((H_MAX/dem.spacing) + 1)
    h_array = np.arange(0,H_MAX+2,dem.spacing)

    c= '%.2f' % round(c, 2)    # Round to two decimal places
    c_file = FDTD_DIR + 'input/' + 'vel_' + FILENAME_ROOT + '.txt'
    cid = open(c_file, 'w')
    for ii in range(0, num_rows):
        cid.write(str(float(h_array[ii])) + ' ' + str(c) + '\n') #why is this a float?
    cid.close()

    rho = '%.2f' % round(rho, 2)    # Round to two decimal places
    rho_file = FDTD_DIR + 'input/' + 'den_' + FILENAME_ROOT + '.txt'
    rhoID = open(rho_file, 'w')
    for ii in range(0, num_rows):
        rhoID.write(str(float(h_array[ii])) + ' ' + rho + '\n')
    rhoID.close()

    print('TOPO_FN = ' + topo_file)
    print('C_FN = ' + c_file)
    print('RHO_FN = ' + rho_file)
    print('STA_FN = ' + str(station_file))
    print('STA_NUM = ' + str(len(station)))

    # loop through every stations and make param file

    sh_name='runall_'+FILENAME_ROOT+'_rtm.sh'
    fsh=open(FDTD_DIR+sh_name,'w')
    fsh.write('#!/bin/sh\n')
    fsh.close()

    if not os.path.isdir(FDTD_DIR + 'input/'):
        os.makedirs(FDTD_DIR + 'input/')

    for i,sta in enumerate(station):
        foutnamenew=FILENAME_ROOT+'_'+sta+'.param'

        #make sure relevant directories exist
        OUTDIRtmp='output_'+sta
        if not os.path.exists(FDTD_DIR+OUTDIRtmp):
            os.makedirs(FDTD_DIR+OUTDIRtmp)

        #see infraFDTD manual for more info!
        f=open(FDTD_DIR+foutnamenew,'w')
        f.write('PATH input=./input output=./'+OUTDIRtmp+'\n')
        f.write('FDMESH x=%d y=%d max_elev=%d dh=%d \n' % (xmax,ymax,H_MAX,dem.spacing))
        f.write('TIME T=%d dt=%.3f\n' % (MAX_T,DT))
        f.write('TOPOGRAPHY elevfile=' + 'elev_' + FILENAME_ROOT + '.txt' + '\n')
        f.write('SOUND_SPEED profile=' + 'vel_' + FILENAME_ROOT + '.txt' + '\n')
        f.write('AIR_DENSITY profile=' + 'den_' + FILENAME_ROOT + '.txt' + '\n')
        #set monopole source at the station!
        f.write('MSOURCE x=%.1f y=%.1f height=0 func=bharris integral=1 freq=%.1f, p0=1\n'
                %(staxyz[i][0],staxyz[i][1],float(SRC_FREQ)))
        f.write('SSNAPSHOT name=sur height=0 interval=%.3f\n'%(SNAPOUT))
        f.write('STATION name=SRC x=%.1f y=%.1f height=0\n'%(staxyz[i][0],staxyz[i][1]))
        f.close()
        print('Saving station file '+foutnamenew)

        #add station onto shell script
        fsh=open(FDTD_DIR+sh_name,'a')
        fsh.write('ifd '+ foutnamenew +' > run_' + FILENAME_ROOT+'_'+sta+'.txt \n')
        fsh.close()


def read_fdtd_files(FDTD_DIR, stations):
    """
    Computes travel time from each station to each grid point using FDTD
    output surface pressure files.

    Args:
        FDTD_DIR: output directory for FDTD run
        stations: SEED station codes (e.g. 'YIFS')
    Returns:
        tprop: [s] propagation times for Stream containing gathered waveforms

    """

    print('--------------')
    print('USING FDTD FILES FOR RTM TIME CALCULATION')
    print('--------------')

    #get surface coordinates and elevations
    indx3=np.loadtxt(FDTD_DIR+ 'output_'+stations[0]+'/sur_coords.txt',
                     dtype = int)

    x=np.unique(indx3[:,0])
    y=np.unique(indx3[:,1])
    nx=len(x)
    ny=len(y)
    nvals=indx3.shape[0]
    nsta=len(stations)

    tprop=np.zeros((nsta,ny,nx))

    #loop through each station and get propagation times to each grid point
    for i,sta in enumerate(stations):
        print('Running for %s'%sta)

        OUTDIRtmp=FDTD_DIR+'output_'+sta+'/'

        #get monopole source time and data vector
        src=np.genfromtxt(OUTDIRtmp+ 'monopole_src_1.txt') #'SRC_wav.txt')
        tvec=src[:,0]
        srcdata=src[:,1]

        #find the delay in the sourc peak
        samax=np.argmax(abs(np.diff(srcdata)))
        srcdelay=tvec[samax-1] #subtract 1 because of the diff

        #get surface snapshot filenames
        fnames=glob.glob(OUTDIRtmp+'sur_pressure*.dat')

        #need to sort by name! this is tricky but seems to work
        fnames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        nfiles=len(fnames)
        print('Reading in %d files for %s and calculating travel times'%(nfiles,OUTDIRtmp))

        #populate surface pressure
        psurf=np.zeros((nfiles,ny,nx))
        for ij,fnametmp in enumerate(fnames):

            f=open(fnametmp,'rb')
            PP0=np.fromfile(f,dtype=np.float64,count=nvals)
            #PP0=np.fromfile(f,dtype=np.float32,count=nvals) #32-bit float for old FDTD version
            f.close()
            psurf[ij,:,:]=np.reshape(PP0,(len(y),len(x)))

        tvec=np.linspace(tvec[0],tvec[-1],nfiles)

        #now determine time delays from each grid point to each station
        for ii in range(ny):
            for jj in range(nx):
                amax=np.argmax(np.abs(psurf[:,ii,jj]))
                tprop[i,ii,jj]=tvec[amax]

        #remove delay from peak of src-time function to get propagation time
        tprop[i,:,:]=tprop[i,:,:]-srcdelay
        print('done\n')

    return tprop


def _generate_travel_time_array(grid, st, method, **method_kwargs):

    # Expand the grid to a 3-D array of (station, y, x)
    travel_times = grid.expand_dims(station=[tr.id for tr in st]).copy()

    if method == 'celerity':

        # Grab this method's keyword arguments
        celerity = method_kwargs['celerity']  # This is required!
        dem = method_kwargs.get('dem')  # This defaults to None if KeyError

        for tr in st:
            for y_coord in grid['y'].values:
                for x_coord in grid['x'].values:

                    if grid.UTM:  # This is a UTM grid

                        # Define x-y coordinate vectors
                        tr_coords = [tr.stats.utm_x, tr.stats.utm_y]
                        grid_coords = [x_coord, y_coord]

                        if dem is not None:
                            # Add the z-coordinates onto the coordinate vectors
                            tr_coords.append(tr.stats.elevation)
                            grid_coords.append(dem.sel(x=x_coord, y=y_coord))

                        # 2-D or 3-D Euclidian distance in meters
                        distance = np.linalg.norm(np.array(tr_coords) -
                                                  np.array(grid_coords))

                    else:  # This is a lat/lon grid
                        # Distance is in meters
                        distance, _, _ = gps2dist_azimuth(y_coord, x_coord,
                                                          tr.stats.latitude,
                                                          tr.stats.longitude)

                    # Store travel time for this station and source grid point
                    # in seconds
                    travel_times.loc[dict(x=x_coord, y=y_coord,
                                          station=tr.id)] = distance / celerity

    else:
        raise NotImplementedError('Only method=\'celerity\' is implemented '
                                  'thus far.')

    return travel_times
