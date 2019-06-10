import json
from obspy import UTCDateTime
from utils import gather_waveforms

# watc_credentials.json contains a single line with format ["user", "password"]
with open('watc_credentials.json') as f:
    watc_username, watc_password = json.load(f)

t1 = UTCDateTime('2016-05-22T07:50:00')
t2 = t1 + 20*60

st = gather_waveforms(source='IRIS', network='TA', station='O20K',
                      starttime=t1, endtime=t2, remove_response=True,
                      watc_username=watc_username, watc_password=watc_password)

#%% Scratch work for now... trying out pre-processing workflow

import matplotlib.pyplot as plt
from obspy.signal.filter import envelope

def plot_st(st, is_downsampled=False):
    fig = plt.figure()
    st.plot(fig=fig)
    fig.axes[0].set_title(st[0].stats.processing[-1], fontsize=8)
    if is_downsampled:
        waveform = fig.axes[0].get_children()[0]
        waveform.set_color('red')
        waveform.set_linewidth(2)
    fig.show()
    return fig

# RAW (with conversion to Pa from data gathering step)

plot_st(st);

# DETREND

std = st.copy()
std.detrend(type='linear')
plot_st(std);

# TAPER

stt = std.copy()
stt.taper(max_percentage=0.05)
plot_st(stt);

# FILTER

FREQ_MIN = 0.5  # [Hz]
FREQ_MAX = 2    # [Hz]
stf = stt.copy()
stf.filter(type='bandpass', freqmin=FREQ_MIN, freqmax=FREQ_MAX, zerophase=True)
plot_st(stf);

# ENVELOPE
ste = stf.copy()
ste[0].data = envelope(ste[0].data)
ste[0].stats.processing.append('obspy.signal.filter.envelope()')
plot_st(ste);

# THREE OPTIONS FOR REDUCING SAMPLING RATE:
# Should we use an anti-aliasing low-pass filter for downsampling?

NEW_SAMPLE_RATE = 0.01  # [Hz]

# (1) RESAMPLE

str1 = ste.copy()

str1.resample(sampling_rate=NEW_SAMPLE_RATE)

fig = plot_st(str1, is_downsampled=True)
ste.plot(fig=fig)
fig.axes[0].set_ylim(top=ste[0].data.max())

# (2) DECIMATE

str2 = ste.copy()

fs = str2[0].stats.sampling_rate
factor = int(fs/NEW_SAMPLE_RATE)
str2.decimate(factor=factor, no_filter=True)  # Fails with no_filter=False

fig = plot_st(str2, is_downsampled=True)
ste.plot(fig=fig)
fig.axes[0].set_ylim(top=ste[0].data.max())

# (3) INTERPOLATE

str3 = ste.copy()

str3.interpolate(sampling_rate=NEW_SAMPLE_RATE)

fig = plot_st(str3, is_downsampled=True)
ste.plot(fig=fig)
fig.axes[0].set_ylim(top=ste[0].data.max())
