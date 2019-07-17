import os
import json
import utm
import numpy as np
import matplotlib.pyplot as plt


def fdtd_rtm_prep(FDTD_DIR,FILENAME_ROOT,station,grid,dem,H_MAX,TEMP,MAX_T,DT,SRC_FREQ,VSNAP,SURSNAP,SNAPOUT):
    """
    Prepare and write RTM/FDTD files. Writes station, elevation, density, and sound speed file. Also
    parameter files for each station and shell script for FDTD calculation 

    Args:
        FDTD_DIR: output directory for FDTD run
        FILENAME_ROOT: FDTD input filename prefix
        station: SEED station code
        grid: xarray.DataArray object containing the grid coordinates and
                  metadata
        dem: 2-D NumPy array of elevation values with identical shape to input
             grid.
        H_MAX: max grid height [m]
        TEMP: temerature for sound speed calculation [K]
        MAX_T: duration of FDTD simulation [s] (make sure it extends across your grid)
        DT: simulation time (dt <= dh/c*np.sqrt(3))
        SRC_FREQ: source frequency [Hz] (make sure at least 20 wavelength per dh)
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
    x=np.array(grid.x-grid.x.min())
    y=np.array(grid.y-grid.y.min())
    xmax=x.max()
    ymax=y.max()
    
    print ('Max_x = ' + str(xmax))
    print('Max_y = ' + str(ymax))
    print('Max Z = '  + str(H_MAX))
    print('Min H = 0')
    print ('dh = ' + str(grid.spacing))
    
    
    # Save DEM into a one-column txt file from lower-left to upper-right, row by row
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
            staxyz[i] = [np.abs(grid.x.values-stautm[i][0]).argmin(),
                  np.abs(grid.y.values-stautm[i][1]).argmin(),staloc[i][2]]
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
        ax.imshow(dem,origin='lower', extent=[min(x), max(x), min(y), max(y)],cmap='jet')
        ax.contour(x,y,dem,line_s,colors='k',linewidths=.35) 
        ax.set_aspect('equal')
        for i,sta in enumerate(station):
            ax.plot(x[staxyz[i][0]],y[staxyz[i][1]],'bo')
            ax.text(x[staxyz[i][0]]+10,y[staxyz[i][1]]+10,sta)
            
    #%% save files for FDTD input
    print('Saving elevation file...%d values'%len(elev))
    f = open(topo_file,'w')
    #elevation header: x,y,dx,dy
    f.write(str(len(x)) + ' ' + str(len(y)) + ' ' + str(float(grid.spacing)) + ' ' + str(float(grid.spacing)) +'\n')
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
        f.write(sta + ' ' + str(temp_x) + ' ' + str(temp_y) + ' ' + str(temp_z) + '\n')
    f.close() 
           
    # create vertical profiles for FDTD. Static values for now
    num_rows = int((H_MAX/grid.spacing) + 1)
    h_array = np.arange(0,H_MAX+2,grid.spacing)
        
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
           
    #%% loop through every stations and make param file
    
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
        f.write('FDMESH x=%d y=%d max_elev=%d dh=%d \n' % (xmax,ymax,H_MAX,grid.spacing))
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



