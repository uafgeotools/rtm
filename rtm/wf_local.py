import numpy as np
from obspy.core import read, Stream
import glob
from obspy import UTCDateTime
import json
#import fnmatch
    

def mseed_local(datadir,network,station,starttime,endtime,time_buffer=0,remove_response=False,return_failed_stations=False):
    """
    Read in waveforms from "local" 1 hr, IRIS-compliant miniseed files, and
    output a Stream object with station/element coordinates attached.
    Optionally remove the sensitivity.

    NOTE:
        Usual RTM usage is to specify a starttime/endtime that brackets the
        estimated source origin time. Then time_buffer is used to download
        enough extra data to account for the time required for an infrasound
        signal to propagate to the farthest station. Because this buffer is so
        critical, this function issues a warning if it remains set to its
        default of 0 s.

    Args:
        datadir: directory where miniseed files live
        network: SEED network code
        station: SEED station code
        starttime: Start time for data request (UTCDateTime)
        endtime: End time for data request (UTCDateTime)
        time_buffer: [s] Extra amount of data to download after endtime
                     (default: 0) (not implemented yet)
        remove_response: conversion to Pa by applying calib. Full response/sensitivity
                    removal not currently implelmented and calib typically
                    applied already in local miniseed files (default: False)                     
        return_failed_stations (in prog): If True, returns a list of station codes that
                                were requested but not downloaded. This
                                disables the standard failed station warning
                                message (default: False) (not implemented yet)
                     
    Returns:
        st_out: Stream containing gathered waveforms
        failed_stations: (Optional) List containing station codes that were
                         requested but not downloaded
    """

    print('--------------')
    print('GATHERING LOCAL MINISEED DATA')
    print('--------------')
    
    #find whole hour to determine number of files
    start_rnd=UTCDateTime(starttime.year,starttime.month,starttime.day,starttime.hour)
    nfiles=int(np.ceil((endtime-start_rnd)/3600))    #find the number of hourly miniseed files

    tstep=3600    #time step in seconds
    ctstep=0
    st_out=Stream()

    #loop through each hour and add data on to existing stream object
    for ii in range(nfiles):
        
        #temporary start and end times
        starttimetmp=starttime+tstep*ctstep
        #endtimetmp=starttimetmp+tstep

        #get miniseed naming strucutre format for each file
        yr=starttimetmp.strftime('%Y')
        hr=starttimetmp.strftime('%H')
        #hr2=endtimetmp.strftime('%H')
        jday = starttimetmp.strftime('%j')

        #now read in miniseed file for each station
        #should we specify a channel and location code? right now I'm saying no
        for sta in station:
            mseed_name=(network+'.'+sta+'*'+ yr + '.' + jday + '.' + hr)
            
            fname=glob.glob(datadir+mseed_name)
            if fname:
                for fnametmp in fname:
                    st_out +=read(fnametmp)    #add data onto existing stream
                    print('Reading in '+fnametmp)
            else:
                print ('\nNo files found for ' + mseed_name)    #debugging here
                print ('Skipping to next station!\n\n')
                continue

        ctstep=ctstep+1
        
    st_out.merge()
    st_out.sort()
    
#    # Check that all requested stations are present in Stream
#    requested_stations = station.split(',')
#    downloaded_stations = [tr.stats.station for tr in st_out]
#    failed_stations = []
#    for sta in requested_stations:
#        # The below check works with wildcards, but obviously cannot dsetect if
#        # ALL stations corresponding to a given wildcard (e.g., O??K) were
#        # downloaded. Thus, if careful station selection is desired, specify
#        # each station explicitly and the below check will then be effective.
#        if not fnmatch.filter(downloaded_stations, sta):
#            if not return_failed_stations:
#                # If we're not returning the failed stations, then show this
#                # warning message to alert the user
#                warnings.warn(f'Station {sta} not downloaded from {source} '
#                              'server for this time period.', RTMWarning)
#            failed_stations.append(sta)

    # If the Stream is empty, then we can stop here
    if st_out.count() == 0:
        print('No data downloaded.')
        if return_failed_stations:
            return st_out, failed_stations
        else:
            return st_out

    # Add zeros to ensure all Traces have same length
    st_out.trim(starttime, endtime, pad=True, fill_value=0)

    # Otherwise, show what the Stream contains
    print(st_out.__str__(extended=True))  # This syntax prints the WHOLE Stream

    #print('Assigning coordinates...')
    with open('watc_infra_coords.json') as f:
        WATC_INFRA_COORDS = json.load(f)
    
    for tr in st_out:
        try:
            tr.stats.latitude, tr.stats.longitude,\
                tr.stats.elevation = WATC_INFRA_COORDS[tr.stats.station]
        except KeyError:
            print(f'No coordinates available for {tr.id}. Stopping.')
            raise

    # Remove sensitivity...typcially already done in local miniseed file (for now!)
    if remove_response:

        print('Removing sensitivity via calib value...')
        for tr in st_out:
            # Just applyin calib for now until we get full response info in! 
            #tr.remove_sensitivity()
            tr.data=tr.data*tr.stats.calib


    return st_out