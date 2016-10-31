import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class GeoPolygon(object):
    def __init__(self):
        self.lonArr=np.array([]);
        self.latArr=np.array([]);
        
    def PlotPolygon(self, inbasemap, alpha=0.7, hatch='x'):
        
        x, y = inbasemap(self.lonArr, self.latArr);
        basinArr=np.append(x,y);
        N=x.size;
        basinArr=basinArr.reshape((2,N));
        basinArr=basinArr.T;
        # poly=Polygon(basinArr, edgecolor= (0.5019607843137255, 0.5019607843137255, 0.5019607843137255),\
        #             alpha=alpha, closed=True, fill=False, hatch=hatch, lw=2);
        # poly=Polygon(basinArr, edgecolor= 'r',\
        #             alpha=None, closed=True, fill=False, hatch=hatch, lw=2)
        poly=Polygon(basinArr, edgecolor= 'k',\
                    alpha=alpha, closed=True, fill=False, hatch=hatch, lw=2)
        # poly=Polygon(basinArr, edgecolor='k',\
        #             alpha=None, closed=True, fill=False, hatch=hatch, lw=2);
        plt.gca().add_patch(poly)
        return

class GeoPolygonLst(object):
    """
    A object contains a list of GeoPolygon
    """
    def __init__(self,geopolygons=None):
        self.geopolygons=[]
        if isinstance(geopolygons, GeoPolygon):
            geopolygons = [geopolygons]
        if geopolygons:
            self.geopolygons.extend(geopolygons)

    def __add__(self, other):
        """
        Add two GeoPolygonLsts with self += other.
        """
        if isinstance(other, GeoPolygon):
            other = GeoPolygonLst([other])
        if not isinstance(other, GeoPolygonLst):
            raise TypeError
        geopolygons = self.geopolygons + other.geopolygons
        return self.__class__(geopolygons=geopolygons)

    def __len__(self):
        """
        Return the number of GeoPolygons in the GeoPolygonLst object.
        """
        return len(self.geopolygons)

    def __getitem__(self, index):
        """
        __getitem__ method of GeoPolygonLst objects.
        :return: GeoPolygon objects
        """
        if isinstance(index, slice):
            return self.__class__(geopolygons=self.geopolygons.__getitem__(index))
        else:
            return self.geopolygons.__getitem__(index)

    def append(self, geopolygon):
        """
        Append a single GeoPolygon object to the current GeoPolygonLst object.
        """
        if isinstance(geopolygon, GeoPolygon):
            self.geopolygons.append(geopolygon)
        else:
            msg = 'Append only supports a single GeoPolygon object as an argument.'
            raise TypeError(msg)
        return self
    
    
    def ReadGeoPolygonLst(self, polygonLst ):
        """
        Read GeoPolygon List from a txt file
        longitude latitude
        """
        f = open(polygonLst, 'r');
        NumbC=0;
        newpolygon=False;
        for lines in f.readlines():
            lines=lines.split();
            if newpolygon==True:
                lon=lines[0];
                if lon=='>':
                    newpolygon=False;
                    self.append(geopolygon)
                    continue;
                else:
                    lon=float(lines[0]);
                    lat=float(lines[1]);
                    geopolygon.lonArr=np.append(geopolygon.lonArr, lon);
                    geopolygon.latArr=np.append(geopolygon.latArr, lat);
            a=lines[0];
            b=lines[1];
            if a=='#' and b!='@P':
                continue;
            if b=='@P':
                NumbC=NumbC+1;
                newpolygon=True;
                geopolygon=GeoPolygon();
                continue;
            f.close()
        print 'End of reading', NumbC, 'geological polygons';
        return
    def PlotPolygon(self, inbasemap, alpha=0.3, hatch='x'):
        for geopolygon in  self.geopolygons:
            geopolygon.PlotPolygon(inbasemap, alpha=alpha, hatch=hatch)
        return
    
    def convert_to_vts(self, outdir='.', Radius=1., name='basin'):
        from tvtk.api import tvtk, write_data
        from sympy.ntheory import primefactors
        lonArr=np.array([]); latArr=np.array([])
        for geopolygon in  self.geopolygons:
            lonArr=np.append(lonArr, geopolygon.lonArr)
            latArr=np.append(latArr, geopolygon.latArr)
        theta=(90.-latArr)/180.*np.pi; phi=lonArr/180.*np.pi
        x = Radius * np.sin(theta) * np.cos(phi)
        y = Radius * np.sin(theta) * np.sin(phi)
        z = Radius * np.cos(theta)
        pts = np.empty(z.shape + (3,), dtype=float)
        pts[..., 0] = x; pts[..., 1] = y; pts[..., 2] = z
        least_prime=primefactors(x.size)[0]
        dims = (x.size/least_prime, least_prime, 1)
        sgrid = tvtk.StructuredGrid(dimensions=dims, points=pts)
        sgrid.point_data.scalars = (np.ones(x.size)).ravel(order='F')
        sgrid.point_data.scalars.name = name
        outfname=outdir+'/'+name+'.vts'
        write_data(sgrid, outfname)
        return