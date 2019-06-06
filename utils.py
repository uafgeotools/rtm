from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.earthworm import Client as EW_Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy import Stream


# Define clients
iris_client = FDSN_Client('IRIS')
avo_client = EW_Client('pubavo1.wr.usgs.gov', port=16023)  # 16023 is long-term


def gather_waveforms(network, station, starttime, endtime):
    """
    Gather waveforms from IRIS or AVO Winston and output a Stream object.

    Args:
        network: SEED network code
        station: SEED station code - no wildcards (?, *) allowed!
        starttime: Start time for data request (UTCDateTime)
        endtime: End time for data request (UTCDateTime)
    Returns:
        st_out: Stream containing gathered waveforms
    """

    # ---------------------------------------------------------------------
    # First, attempt to obtain data from IRIS...
    # ---------------------------------------------------------------------
    try:
        st_out = iris_client.get_waveforms(network, station, '*', 'BDF,HDF',
                                           starttime, endtime)

    # ---------------------------------------------------------------------
    # ...if that fails, try AVO Winston
    # ---------------------------------------------------------------------
    except FDSNNoDataException:
        print('No data found at IRIS. Trying AVO Winston...')

        # ---------------------------------------------------------------------
        # Array case
        # ---------------------------------------------------------------------
        if station in ['ADKI', 'AKS', 'DLL', 'OKIF', 'SDPI']:

            # Select the correct channel
            if station in ['DLL', 'OKIF']:
                channel = 'HDF'
            else:
                channel = 'BDF'

            st_out = Stream()  # Make an empty Stream object to populate

            # Deal with funky channel naming convention for AKS (for all other
            # arrays, six elements are assumed)
            if station == 'AKS':
                for channel in ['BDF', 'BDG', 'BDH', 'BDI']:
                    st_out += avo_client.get_waveforms(network, station, '',
                                                       channel, starttime,
                                                       endtime)
            else:
                for location in ['01', '02', '03', '04', '05', '06']:
                    st_out += avo_client.get_waveforms(network, station,
                                                       location, channel,
                                                       starttime, endtime)

        # ---------------------------------------------------------------------
        # Single station case
        # ---------------------------------------------------------------------
        else:
            st_out = avo_client.get_waveforms(network, station, '', 'BDF',
                                              starttime, endtime)

    st_out.sort()

    return st_out
