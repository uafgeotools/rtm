import json
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.earthworm import Client as EW_Client
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from obspy import Stream


# Load AVO infrasound station calibration values (units are Pa/ct)
AVO_INFRA_CALIB_FILE = 'avo_infra_calib_vals.json'
with open(AVO_INFRA_CALIB_FILE) as f:
    avo_calib_values = json.load(f)

# Define IRIS and AVO clients (define WATC client within function)
iris_client = FDSN_Client('IRIS')
avo_client = EW_Client('pubavo1.wr.usgs.gov', port=16023)  # 16023 is long-term


def gather_waveforms(source, network, station, starttime, endtime,
                     remove_response=False, watc_username=None,
                     watc_password=None):
    """
    Gather infrasound waveforms from IRIS or WATC FDSN, or AVO Winston, and
    output a Stream object with station/element coordinates attached.
    Optionally remove the sensitivity.

    Args:
        source: Which source to gather waveforms from - options are:
                'IRIS' <-- IRIS FDSN
                'WATC' <-- WATC FDSN
                'AVO'  <-- AVO Winston
        network: SEED network code
        station: SEED station code
        starttime: Start time for data request (UTCDateTime)
        endtime: End time for data request (UTCDateTime)
        remove_response: Toggle conversion to Pa via remove_sensitivity() if
                         available, else just do a simple scalar multiplication
        watc_username: Username for WATC FDSN server
        watc_password: Password for WATC FDSN server
    Returns:
        st_out: Stream containing gathered waveforms
    """

    print('--------------')
    print('GATHERING DATA')
    print('--------------')

    # IRIS FDSN
    if source == 'IRIS':

        print('Reading data from IRIS FDSN...')
        st_out = iris_client.get_waveforms(network, station, '*', 'BDF,HDF',
                                           starttime, endtime,
                                           attach_response=remove_response)

    # WATC FDSN
    elif source == 'WATC':

        print('Connecting to WATC FDSN...')
        try:
            watc_client = FDSN_Client('http://10.30.5.10:8080',
                                      user=watc_username,
                                      password=watc_password)
        except FDSNException:
            print('Issue connecting to WATC FDSN. Check your VPN '
                  'connection and try again.')
            return Stream()

        print('Successfully connected. Reading data from WATC FDSN...')
        st_out = watc_client.get_waveforms(network, station, '*', 'BDF,HDF',
                                           starttime, endtime,
                                           attach_response=remove_response)

    # AVO Winston
    elif source == 'AVO':

        print('Reading data from AVO Winston...')

        # Array case
        if station in ['ADKI', 'AKS', 'DLL', 'OKIF', 'SDPI']:

            # Select the correct channel
            if station in ['DLL', 'OKIF']:
                channel = 'HDF'
            else:
                channel = 'BDF'

            st_out = Stream()  # Make an empty Stream object to populate

            # Deal with funky channel naming convention for AKS (for all other
            # arrays, six numbered elements are assumed)
            if station == 'AKS':
                for channel in ['BDF', 'BDG', 'BDH', 'BDI', 'BDJ', 'BDK']:
                    st_out += avo_client.get_waveforms(network, station, '--',
                                                       channel, starttime,
                                                       endtime)
            else:
                for location in ['01', '02', '03', '04', '05', '06']:
                    st_out += avo_client.get_waveforms(network, station,
                                                       location, channel,
                                                       starttime, endtime)

        # Single station case
        else:
            st_out = avo_client.get_waveforms(network, station, '--', 'BDF',
                                              starttime, endtime)

            # Special case for CLES1 and CLES2 which also have HDF channels
            if station in ['CLES1', 'CLES2']:
                st_out += avo_client.get_waveforms(network, station, '--',
                                                   'HDF', starttime, endtime)

    else:

        print('Unrecognized source. Valid options are \'IRIS\', \'WATC\', or '
              '\'AVO\'.')
        return Stream()

    # Add zeros to ensure all Traces have same length
    st_out.trim(starttime, endtime, pad=True, fill_value=0)

    st_out.sort()

    print(st_out)

    print('---------------------')
    print('ASSIGNING COORDINATES')
    print('---------------------')

    # Assign coordinates using IRIS FDSN regardless of data source
    try:
        inv = iris_client.get_stations(network=network, station=station,
                                       starttime=starttime, endtime=endtime,
                                       level='channel')
    except FDSNNoDataException:
        inv = []

    for tr in st_out:
        for nw in inv:
            for sta in nw:
                for cha in sta:
                    # Being very thorough to check if everything matches!
                    if (tr.stats.network == nw.code and
                            tr.stats.station == sta.code and
                            tr.stats.location == cha.location_code and
                            tr.stats.channel == cha.code):

                        tr.stats.longitude = cha.longitude
                        tr.stats.latitude = cha.latitude

    # Report if any Trace did NOT get coordinates assigned
    print('Traces WITHOUT coordinates assigned:')
    num_unassigned = 0
    for tr in st_out:
        try:
            tr.stats.longitude, tr.stats.latitude
        except AttributeError:
            print('    ' + tr.id)
            num_unassigned += 1
    if num_unassigned == 0:
        print('    None')

    # Remove sensitivity
    if remove_response:

        print('--------------------')
        print('REMOVING SENSITIVITY')
        print('--------------------')

        unremoved_ids = []
        for tr in st_out:
            print(tr.id)
            try:
                # Just removing sensitivity for now. remove_response() can lead
                # to errors. This should be sufficient for now. Plus some
                # IRIS-AVO responses are wonky.
                tr.remove_sensitivity()
                print('    Sensitivity removed using attached response.')
            except ValueError:
                print('    No response information available.')
                try:
                    calib = avo_calib_values[tr.stats.station]
                    tr.data = tr.data * calib
                    tr.stats.processing.append('Data multiplied by '
                                               f'calibration value of {calib} '
                                               'Pa/ct')
                    print('    Sensitivity removed using calibration value of '
                          f'{calib} Pa/ct.')
                except KeyError:
                    print('    No calibration value available.')
                    unremoved_ids.append(tr.id)

        # Report if any Trace did NOT get sensitivity removed
        print('Traces WITHOUT sensitivity removed:')
        [print('    ' + tr_id) for tr_id in unremoved_ids]
        if len(unremoved_ids) == 0:
            print('    None')

    return st_out
