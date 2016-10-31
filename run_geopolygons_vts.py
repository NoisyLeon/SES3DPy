import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
import GeoPolygon


basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')
basins.convert_to_vts()