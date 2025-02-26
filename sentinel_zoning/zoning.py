from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union
from shapely import wkt
import shapely
#import rasterio
import numpy as np
#from rasterio.warp import transform
from pyproj import CRS, Transformer
import mapclassify

class Zoning:
    def __init__(self, sentinel):
        self._sentinel = sentinel

    def zone(self, dates, west, east, south, north, number_of_zones, multi_polygon_wkt):
        # create bounding polygon
        multipolygon = unary_union(wkt.loads(multi_polygon_wkt))

        # gather ndvi information
        geoattrs = self._sentinel.get_geoattrs(west, east, south, north)
        ndvis = []
        for date in dates:
            ndvis += [self._sentinel.get_index(date, 'NDVI', west, east, south, north, save_result_as_tiff=True)]
        if len(ndvis) < 1:
            raise Exception(f"unable to get ndvis for any of the dates: {dates}")

        # create individual coordinates for each point of the array in an array of the same size
        xs = np.linspace(geoattrs['bounds'].left, geoattrs['bounds'].right, geoattrs['shape'][1])
        ys = np.linspace(geoattrs['bounds'].top, geoattrs['bounds'].bottom, geoattrs['shape'][0])
        xgrid, ygrid = np.meshgrid(xs, ys)
        crs = CRS.from_user_input(geoattrs['crs'])
        transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(xgrid, ygrid)
        coords = np.stack((lon, lat), axis=-1)

        # pad so no out of bounds occur
        coords = np.pad(coords, pad_width=((1,1),(1,1),(0,0)), mode='constant', constant_values=0)
        # calculate first and last rows excluding corners
        coords[0, 1:-1, 0] = 2 * coords[1, 1:-1, 0] - coords[2, 1:-1, 0]
        coords[0, 1:-1, 1] = 2 * coords[1, 1:-1, 1] - coords[2, 1:-1, 1]
        coords[-1, 1:-1, 0] = 2 * coords[-2, 1:-1, 0] - coords[-3, 1:-1, 0]
        coords[-1, 1:-1, 1] = 2 * coords[-2, 1:-1, 1] - coords[-3, 1:-1, 1]
        # calculate first and last columns
        coords[:, 0, 0] = 2 * coords[:, 1, 0] - coords[:, 2, 0]
        coords[:, 0, 1] = 2 * coords[:, 1, 1] - coords[:, 2, 1]
        coords[:, -1, 0] = 2 * coords[:, -2, 0] - coords[:, -3, 0]
        coords[:, -1, 1] = 2 * coords[:, -2, 1] - coords[:, -3, 1]

        # create mask array with values inside polygon, other values are -1
        masked_ndvis = []
        for ndvi in ndvis:
            masked_ndvis += [self._masked_array(multipolygon, coords, ndvi)]
        masked_ndvis = np.stack(masked_ndvis)
        average_ndvi = np.mean(masked_ndvis, axis=0)

        # create zones (-1 for no zone)
        #zones = np.full((average_ndvi.shape), -1, dtype=np.int16)
        #_min = np.min(average_ndvi[average_ndvi >= 0])
        #_max = np.max(average_ndvi[average_ndvi >= 0])
        ##_distribution = [0, 0.4, 0.55, 0.7, 0.85]
        #_distribution = [0, 0.3, 0.5, 0.66, 0.83]
        #_boundaries = [_min + (_max - _min) * dist for dist in _distribution]
        ##_step = (_max - _min) / number_of_zones
        #for i in range(0, number_of_zones):
        #    #zones[average_ndvi >= _min + i * _step] = i
        #    zones[average_ndvi >= _boundaries[i]] = i
        #zones = np.pad(zones, pad_width=1, mode='constant', constant_values=-1)

        # create zones (-1 for no zone) using natural breaks
        natural_breaks = mapclassify.NaturalBreaks(average_ndvi[average_ndvi >= 0], k=number_of_zones)
        zones = np.full(average_ndvi.shape, -1, dtype=np.int16)
        zones[average_ndvi >= 0] = natural_breaks.yb
        # pad to avoid oob error
        zones = np.pad(zones, pad_width=1, mode='constant', constant_values=-1)

        # test save to tiff file
        #self._sentinel.uint16_array_to_tiff((zones + 1) * 10000, 'test2', f"{dates[0]}-{dates[-1]}", 'L2A', 'Zones', zones.shape[0], zones.shape[1], geoattrs['crs'], geoattrs['transform'])
        return self._zones_to_polygon_wkts(number_of_zones, zones, coords, multipolygon)

    def _polygon_around_point(self, coords, y, x):
        '''Polygon um einen Punkt herum bis zur Hälfte des nächsten Punktes, leicht größer, damit Rundungsfehler die spätere Vereinigung zu einem großen Polygon nicht verhindern'''
        def _centroid(points, weights):
            return (sum(p[0] * w for p, w in zip(points, weights)) / sum(weights)), (sum(p[1] * w for p, w in zip(points, weights)) / sum(weights))
        # slightly increased weights for the diagonal point to overlap the polygons a little bit for building the union
        _weight_offset = 0.000001
        return Polygon([
            _centroid([coords[y-1][x-1], coords[y-1][x], coords[y][x], coords[y][x-1]], [1 + _weight_offset, 1, 1, 1]),
            _centroid([coords[y-1][x], coords[y-1][x+1], coords[y][x+1], coords[y][x]], [1, 1 + _weight_offset, 1, 1]),
            _centroid([coords[y][x], coords[y][x+1], coords[y+1][x+1], coords[y+1][x]], [1, 1, 1 + _weight_offset, 1]),
            _centroid([coords[y][x-1], coords[y][x], coords[y+1][x], coords[y+1][x-1]], [1, 1, 1, 1 + _weight_offset])
        ])

    def _masked_array(self, multipolygon, padded_coords_array, values_array):
        ret = np.full((values_array.shape), -1, dtype=np.int32)
        for y in range(values_array.shape[0]):
            for x in range(values_array.shape[1]):
                if multipolygon.intersects(self._polygon_around_point(padded_coords_array, y + 1, x + 1)):
                    ret[y, x] = values_array[y, x]
        return ret

    def _zones_to_polygon_wkts(self, number_of_zones, zones, coords, bounding_multipolygon):
        # calculate polygons for each pixel
        polygons = [[] for _ in range(number_of_zones)]
        for y in range(1, zones.shape[0] - 1):
            for x in range(1, zones.shape[1] - 1):
                if zones[y][x] < 0:
                    continue
                polygons[zones[y][x]] += [self._polygon_around_point(coords, y, x)]
                #print(f"{shapely.to_wkt(polygons[zones[y][x]][-1], rounding_precision=12)},")
        # make union
        ret = {}
        for i in range(number_of_zones):
            if polygons[i]:
                union = unary_union(MultiPolygon(polygons[i])).intersection(bounding_multipolygon)
                if union.geom_type == 'Polygon':
                    ret[i] = [shapely.to_wkt(union, rounding_precision=8)]
                else:
                    ret[i] = [shapely.to_wkt(poly, rounding_precision=8) for poly in union.geoms]
        return ret
