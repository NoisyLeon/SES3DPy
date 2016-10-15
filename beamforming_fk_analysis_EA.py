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
st.append(dset.get_trace(staid=None, lon=110.25, lat=33.75, outdir='.', SLst=SLst, channel='BXZ'))
st.append(dset.get_trace(staid=None, lon=109.75, lat=34.25, outdir='.', SLst=SLst, channel='BXZ'))
st.append(dset.get_trace(staid=None, lon=110.25, lat=34.25, outdir='.', SLst=SLst, channel='BXZ'))
st.append(dset.get_trace(staid=None, lon=109.75, lat=33.75, outdir='.', SLst=SLst, channel='BXZ'))

stime=obspy.UTCDateTime(0)+620
etime=stime+100


print 'Start fk analysis'

# Execute array_processing
kwargs = dict(
    # slowness grid: X min, X max, Y min, Y max, Slow Step
    sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
    # sliding window properties
    win_len=5.0, win_frac=0.2,
    # frequency properties
    frqlow=0.08, frqhigh=.125, prewhiten=0,
    # restrict output
    semb_thres=-1e9, vel_thres=-1e9,
    stime=stime,
    etime=etime,
    method=1
)
out = array_processing(st, **kwargs)
print 'End fk'
# Plot

cmap = obspy_sequential

# make output human readable, adjust backazimuth to values between 0 and 360
t, rel_power, abs_power, baz, slow = out.T
baz[baz < 0.0] += 360

# choose number of fractions in plot (desirably 360 degree/N is an integer!)
N = 36
N2 = 30
abins = np.arange(N + 1) * 360. / N
sbins = np.linspace(0, 3, N2 + 1)

# sum rel power in bins given by abins and sbins
hist, baz_edges, sl_edges = \
    np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)

# transform to radian
baz_edges = np.radians(baz_edges)

# add polar and colorbar axes
fig = plt.figure(figsize=(8, 8))
cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")

dh = abs(sl_edges[1] - sl_edges[0])
dw = abs(baz_edges[1] - baz_edges[0])

# circle through backazimuth
for i, row in enumerate(hist):
    bars = ax.bar(left=(i * dw) * np.ones(N2),
                  height=dh * np.ones(N2),
                  width=dw, bottom=dh * np.arange(N2),
                  color=cmap(row / hist.max()))

ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
ax.set_xticklabels(['N', 'E', 'S', 'W'])

# set slowness limits
ax.set_ylim(0, 3)
[i.set_color('grey') for i in ax.get_yticklabels()]
ColorbarBase(cax, cmap=cmap,
             norm=Normalize(vmin=hist.min(), vmax=hist.max()))

plt.show()
