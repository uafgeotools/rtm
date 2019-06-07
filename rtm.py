import json
from obspy import UTCDateTime
from utils import gather_waveforms

# watc_credentials.json contains a single line with format ["user", "password"]
with open('watc_credentials.json') as f:
    watc_username, watc_password = json.load(f)

t1 = UTCDateTime('2018-08-22T07:58')
t2 = t1 + 60

st = gather_waveforms(source='AVO', network='AV', station='SDPI',
                      starttime=t1, endtime=t2, remove_response=True,
                      watc_username=watc_username, watc_password=watc_password)
