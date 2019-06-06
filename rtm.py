from obspy import UTCDateTime
from utils import gather_waveforms

t1 = UTCDateTime('2018-08-22T07:58')
t2 = t1 + 60

st = gather_waveforms('AV', 'SDPI', t1, t2)
print(st)
