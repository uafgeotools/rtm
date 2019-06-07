import json
from obspy import UTCDateTime
from utils import gather_waveforms

# watc_credentials.json contains a single line with format ["user", "password"]
with open('watc_credentials.json') as f:
    watc_username, watc_password = json.load(f)

t1 = UTCDateTime('2019-06-07T07:58:00')
t2 = t1 + 60

sta='I53H*'

st = gather_waveforms(source='IRIS', network='IM', station=sta,
                      starttime=t1, endtime=t2, remove_response=True,
                      watc_username=watc_username, watc_password=watc_password)
