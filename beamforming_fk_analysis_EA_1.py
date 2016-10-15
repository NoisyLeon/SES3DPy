import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

import obspy
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.invsim import corn_freq_2_paz
from obspy.signal.array_analysis import array_processing


import symdata
import stations
import obspy.signal.filter
import numpy as np
stafile='/work3/leon/ASDF_data/recfile_1'
dset = symdata.ses3dASDF('/work3/leon/ASDF_data/ses3d_2016.h5')

SLst=stations.StaLst()
SLst.read(stafile)
st=obspy.Stream()
st.append(dset.get_trace(staid=None, lon=110, lat=34., outdir='.', SLst=SLst, channel='BXZ'))
st.append(dset.get_trace(staid=None, lon=110, lat=34.25, outdir='.', SLst=SLst, channel='BXZ'))
st.append(dset.get_trace(staid=None, lon=110, lat=33.75, outdir='.', SLst=SLst, channel='BXZ'))
st.append(dset.get_trace(staid=None, lon=110.25, lat=34., outdir='.', SLst=SLst, channel='BXZ'))
st.append(dset.get_trace(staid=None, lon=109.75, lat=34., outdir='.', SLst=SLst, channel='BXZ'))

stime=obspy.UTCDateTime(0)+600
etime=stime+100


print 'Start fk analysis'
# Execute array_processing
kwargs = dict(
    # slowness grid: X min, X max, Y min, Y max, Slow Step
    sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.05,
    # sliding window properties
    win_len=5.0, win_frac=0.1,
    # frequency properties
    frqlow=0.08, frqhigh=0.125, prewhiten=0,
    # restrict output
    semb_thres=-1e9, vel_thres=-1e9,
    stime=stime,
    etime=etime,
    method=1
)
out = array_processing(st, **kwargs)
print 'End fk analysis'
# Plot

# Plot
labels = ['rel.power', 'abs.power', 'baz', 'slow']

xlocator = mdates.AutoDateLocator()
fig = plt.figure()
for i, lab in enumerate(labels):
    ax = fig.add_subplot(4, 1, i + 1)
    ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
               edgecolors='none', cmap=obspy_sequential)
    ax.set_ylabel(lab)
    ax.set_xlim(out[0, 0], out[-1, 0])
    ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
    ax.xaxis.set_major_locator(xlocator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

fig.suptitle('AGFA skyscraper blasting in Munich %s' % (
    stime.strftime('%Y-%m-%d'), ))
fig.autofmt_xdate()
fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
plt.show()
