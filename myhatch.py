"""
Hatching (pattern filled polygons) is supported currently in the PS,
PDF, SVG and Agg backends only.
"""
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Polygon
import numpy as np
fig= plt.figure()
ax = fig.add_subplot(111)
basin=np.loadtxt('basin_test')
blst=[];
lon=basin[:,0]
lat=basin[:,1]
N=lon.size
verts = []
codes=[]
for i in np.arange(N):
    verts.append((lon[i], lat[i]))
    if i==0:
        codes.append(Path.MOVETO);
    elif i==N-1:
        codes.append(Path.CLOSEPOLY);
    else:
        codes.append(Path.CURVE4);


path = Path(verts, codes)

poly=Polygon(basin, closed=True,
                      fill=False, hatch='x')
fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, lw=2)
ax.add_patch(poly)
ax.set_xlim((110, 130));
ax.set_ylim((50, 56))
plt.show()
