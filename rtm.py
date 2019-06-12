#%% (1) Grab the data

import json
from obspy import UTCDateTime
from utils import gather_waveforms, process_waveforms

# watc_credentials.json contains a single line with format ["user", "password"]
with open('watc_credentials.json') as f:
    watc_username, watc_password = json.load(f)

t1 = UTCDateTime('2016-05-22T07:45:00')
t2 = t1 + 40*60

st = gather_waveforms(source='IRIS', network='AK,TA',
                      station='HOM,M19K,M22K,O20K,O22K,RC01', starttime=t1,
                      endtime=t2, remove_response=True,
                      watc_username=watc_username, watc_password=watc_password)

#%% (2) Process the data

INTERP_RATE = 0.05    # [Hz] New sampling rate to interpolate to

FREQ_MIN = 0.5        # [Hz] Lower bandpass corner
FREQ_MAX = 2          # [Hz] Upper bandpass corner

AGC_WIN = 250         # [s] Window for AGC
AGC_METHOD = 'gismo'  # Method to use for AGC

agc_params = dict(win_sec=AGC_WIN, method=AGC_METHOD)

st_proc = process_waveforms(st, interp_rate=INTERP_RATE, freqmin=FREQ_MIN,
                            freqmax=FREQ_MAX, agc_params=agc_params,
                            normalize=True, plot_steps=True)
