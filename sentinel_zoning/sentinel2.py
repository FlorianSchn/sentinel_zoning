import openeo
import json
use_rasterio = True
if use_rasterio:
    import rasterio
else:
    # evtl. funktioniert gdal in Arcgis direkt, die untenstehenden gdal Codes sind aber nicht getestet
    from osgeo import gdal, osr
import numpy as np
import os
import glob
import re
from pyproj import CRS, Transformer
from datetime import datetime, timedelta
from s2cloudless import S2PixelCloudDetector

class Sentinel2:
    def __init__(self, openeo_url, output_dir, force_auth):
        self._openeo_url = openeo_url
        self._output_dir = output_dir
        self._connected = False
        # TODO Authentifizierung erneuern, wenn z.B. Credits abgelaufen
        self._force_auth = force_auth
        if not use_rasterio:
            gdal.UseExceptions()
        # TODO evtl. nur die Bands herunterladen, die benötigt werden
        #self._bands_active = list(set(active_bands + )) # necessary bands for s2cloudless algorithm
        self.bands = {
            "L1C": {
                # name: (resolution, min, max, description)
                'B01': (60, 1, 10000, 'Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)'),
                'B02': (10, 1, 10000, 'Blue, 492.4 nm (S2A), 492.1 nm (S2B)'),
                'B03': (10, 1, 10000, 'Green, 559.8 nm (S2A), 559.0 nm (S2B)'),
                'B04': (10, 1, 10000, 'Red, 664.6 nm (S2A), 665.0 nm (S2B)'),
                'B05': (20, 1, 10000, 'Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)'),
                'B06': (20, 1, 10000, 'Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)'),
                'B07': (20, 1, 10000, 'Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)'),
                'B08': (10, 1, 10000, 'NIR, 832.8 nm (S2A), 833.0 nm (S2B)'),
                'B8A': (20, 1, 10000, 'Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)'),
                'B09': (60, 1, 10000, 'Water vapour, 945.1 nm (S2A), 943.2 nm (S2B)'),
                'B10': (60, 1, 10000, 'SWIR, 1375 nm'),
                'B11': (20, 1, 10000, 'SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)'),
                'B12': (20, 1, 10000, 'SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)'),
            }, "L2A": {
                # name: (resolution, min, max, description)
                'B01': (60, 1, 10000, 'Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)'),
                'B02': (10, 1, 10000, 'Blue, 492.4 nm (S2A), 492.1 nm (S2B)'),
                'B03': (10, 1, 10000, 'Green, 559.8 nm (S2A), 559.0 nm (S2B)'),
                'B04': (10, 1, 10000, 'Red, 664.6 nm (S2A), 665.0 nm (S2B)'),
                'B05': (20, 1, 10000, 'Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)'),
                'B06': (20, 1, 10000, 'Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)'),
                'B07': (20, 1, 10000, 'Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)'),
                'B08': (10, 1, 10000, 'NIR, 832.8 nm (S2A), 833.0 nm (S2B)'),
                'B8A': (20, 1, 10000, 'Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)'),
                'B09': (60, 1, 10000, 'Water vapour, 945.1 nm (S2A), 943.2 nm (S2B)'),
                'B11': (20, 1, 10000, 'SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)'),
                'B12': (20, 1, 10000, 'SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)'),
                'AOT': (20, 1, 65535, 'Aerosol optical thickness'),
                'WVP': (20, 1, 65535, 'Scene average water vapour'),
                'SCL': (20, 0, 255, 'Scene classification layer'),
            }
        }

    def _connect(self):
        '''connect to openeo host'''
        if not self._connected:
            print(f"connecting to {self._openeo_url}")
            self._con = openeo.connect(self._openeo_url)
            self._con.authenticate_oidc()
            self._connected = True

    def provide(self, west, east, south, north, start, end):
        '''stellt sicher, dass die Daten für die gewählte Eingabe lokal verfügbar sind'''
        spatial_extent = {"west": west, "east": east, "south": south, "north": north}
        subfolder = self._get_subfolder(west, east, south, north)
        if subfolder:
            # nur Daten herunterladen, die noch nicht heruntergeladen sind
            temporal_extents = self._missing_temporal_extents(subfolder, start, end)
        else:
            temporal_extents = [[start, end]]
        if len(temporal_extents) > 0:
            self._connect()
        for temporal_extent in temporal_extents:
            for level in self.bands.keys():
                print(f"downloading sentinel2-{level.lower()} in {temporal_extent}")
                cube = self._con.load_collection(f"SENTINEL2_{level}", spatial_extent=spatial_extent, temporal_extent=temporal_extent, max_cloud_cover=100, bands=list(self.bands[level].keys()))
                cube = cube.save_result(format='json')
                # hier erfolgt der eigentliche Download
                json = cube.execute()
                # Download in tiffs speichern und metadaten aktualisieren
                self._json_to_tiffs(json, level)
                self._update_meta(json, west, east, south, north, temporal_extent[0], temporal_extent[1])

    def get_cached_timezones(self, west, east, south, north):
        '''die gemerkten Zeiträume zurückgeben'''
        try:
            with open(os.path.join(self._output_dir, "_meta.json"), "r") as f:
                data = json.load(f)
            return data[f"{west:.8f},{south:.8f}-{east:.8f},{north:.8f}"]["cached_timezones"]
        except:
            return []

    def set_cached_timezones(self, west, east, south, north, timezones):
        '''Zeiträume für die nächste Ausführung merken'''
        try:
            with open(os.path.join(self._output_dir, "_meta.json"), "r") as f:
                data = json.load(f)
            data[f"{west:.8f},{south:.8f}-{east:.8f},{north:.8f}"]["cached_timezones"] = timezones
            with open(os.path.join(self._output_dir, "_meta.json"), "w") as f:
                json.dump(data, f, indent=4)
        except:
            pass

    def subfolder_name(self, xvals, yvals, wkt):
        '''
        erstellt einen eindeutigen Namen für einen Ordner zum Speichern für die eingegebenen Koordinaten
        die Koordinaten werden auf 8 Dezimalstellen gerundet, das sind etwas über 1mm
        '''
        crs = CRS.from_wkt(wkt)
        transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        min_coords = (min(xvals), min(yvals))
        max_coords = (max(xvals), max(yvals))
        min_lon, min_lat = transformer.transform(*min_coords)
        max_lon, max_lat = transformer.transform(*max_coords)
        return f"{min_lon:.8f},{min_lat:.8f}-{max_lon:.8f},{max_lat:.8f}"

    def _get_subfolder(self, west, east, south, north):
        '''den Ordnernamen für die Dateien für ein Gebiet finden'''
        try:
            # _meta.json maps requested spatial extents to sentinel2 spatial extents
            with open(os.path.join(self._output_dir, "_meta.json"), "r") as f:
                data = json.load(f)
            return data[f"{west:.8f},{south:.8f}-{east:.8f},{north:.8f}"]["subfolder"]
        except:
            return ""

    def uint16_array_to_tiff(self, array, subfolder, date, level, band, height, width, crs, transform, photometric=""):
        '''Pixel- und Geodaten in einer Geotiff Datei speichern'''
        output_dir = os.path.join(self._output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{date}-{level}-{band}.tif")
        if use_rasterio:
            if photometric == 'rgb':
                with rasterio.open(output_filename, 'w', driver='GTiff', height=height, width=width, count=3, dtype=np.uint16, crs=crs, transform=transform, compress='lzw', photometric='rgb') as f:
                    f.write(array)
                    f.update_tags(AREA_OR_POINT='Area')
            else:
                with rasterio.open(output_filename, 'w', driver='GTiff', height=height, width=width, count=1, dtype=np.uint16, crs=crs, transform=transform, compress='lzw') as f:
                    f.write(array, 1)
                    f.update_tags(AREA_OR_POINT='Area')
        else:
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(output_filename, width, height, 3 if photometric == 'rgb' else 1, gdal.GDT_UInt16, options=['COMPRESS=LZW'] + ['PHOTOMETRIC=RGB'] if photometric == 'rgb' else [])
            out_ds.SetGeoTransform(transform)
            out_ds.SetProjection(crs.to_wkt())
            if photometric == 'rgb':
                out_ds.GetRasterBand(1).WriteArray(array[0])
                out_ds.GetRasterBand(2).WriteArray(array[1])
                out_ds.GetRasterBand(3).WriteArray(array[2])
            else:
                out_ds.GetRasterBand(1).WriteArray(array)
            out_ds.SetMetadataItem('AREA_OR_POINT', 'Area')
            out_ds = None
        #print(f"created {output_filename}")

    def array_from_tiff(self, subfolder, date, level, band, dtype):
        '''lädt Pixeldaten einer Geotiff Datei'''
        input_filename = os.path.join(self._output_dir, subfolder, f"{date}-{level}-{band}.tif")
        if use_rasterio:
            try:
                with rasterio.open(input_filename) as src:
                    arr = src.read(1)
            except rasterio.errors.RasterioError as e:
                raise Exception(f"cannot open {input_filename}, error {e}")
        else:
            try:
                ds = gdal.Open(input_filename)
                if ds is None:
                    raise Exception(f"cannot open {input_filename}")
                arr = ds.GetRasterBand(1).ReadAsArray()
            except Exception as e:
                raise Exception(f"cannot open {input_filename}, error: {e}")
            finally:
                ds = None
        if dtype == 'uint16':
            return arr
        elif dtype == 'float':
            arr = arr.astype(np.float32) / 65535.0
            return np.clip(arr, 0., 1.)
        raise Exception(f"dtype {dtype} not supported")
        
    def _band_value_to_uint16(self, level, band, value):
        '''normalisiert die Daten der Bänder in den Bereich 0-65535 (also den Bereich von uin16)'''
        if band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']:
            _min = self.bands[level][band][1]
            _max = self.bands[level][band][2]
            _newmin = 0
            _newmax = 65535
            return round(((value - _min) / (_max - _min)) * (_newmax - _newmin) + _newmin)
        else:
            return value

    def _json_to_tiffs(self, json, level, make_rgb=False):
        '''erstellt Geotiff Dateien mit den heruntergeladenen Daten'''
        # extract relevant information
        bands = json["coords"]["bands"]["data"]
        timestamps = json["coords"]["t"]["data"]
        width = json["coords"]["x"]["attrs"]["shape"][0]
        height = json["coords"]["y"]["attrs"]["shape"][0]
        if use_rasterio:
            transform = rasterio.transform.from_bounds(
                json["coords"]["x"]["data"][0],
                json["coords"]["y"]["data"][0],
                json["coords"]["x"]["data"][-1],
                json["coords"]["y"]["data"][-1],
                width, height)
        else:
            left = json["coords"]["x"]["data"][0]
            bottom = json["coords"]["y"]["data"][0]
            right = json["coords"]["x"]["data"][-1]
            top = json["coords"]["y"]["data"][-1]
            pixel_width = (right - left) / width
            pixel_height = (top - bottom) / height
            transform = (left, pixel_width, 0, top, 0, -pixel_height)
        crs = json["attrs"]["crs"]
        dtype = json["attrs"]["dtype"]
        xvals = json["coords"]["x"]["data"]
        yvals = json["coords"]["y"]["data"]
        wkt = json["coords"]["spatial_ref"]["attrs"]["crs_wkt"]

        # create tiff files
        for band_index, band in enumerate(bands):
            for timestamp_index, timestamp in enumerate(timestamps):
                pixel_data = np.asarray(json["data"][timestamp_index][band_index], dtype=np.dtype(dtype))
                # Orientierung fixen
                pixel_data = np.transpose(pixel_data)
                pixel_data = np.flipud(pixel_data)
                # Werte normalisieren
                band_value_uint16 = np.vectorize(lambda value: self._band_value_to_uint16(level, band, value))
                pixel_data = band_value_uint16(pixel_data).astype('uint16')
                # Datei speichern
                self.uint16_array_to_tiff(pixel_data, self.subfolder_name(xvals, yvals, wkt), timestamp.split()[0], level, band, height, width, crs, transform)

    def _date_add(self, date, days):
        '''fügt "days" Tage zu einem Datum hinzu oder zieht sie ab wenn "days" negativ'''
        _date = datetime.strptime(date, '%Y-%m-%d')
        _date += timedelta(days=days)
        return _date.strftime('%Y-%m-%d')

    def _missing_temporal_extents(self, subfolder, start, end):
        '''
        berechnet noch nicht heruntergeladene Zeiträume innerhalb eines durch start und end vorgegebenen Zeitraums
        TODO überprüfen ob die Berechnung wirklich richtig ist
        '''
        try:
            # heruntergeladene Zeiträume einlesen
            with open(os.path.join(self._output_dir, subfolder, "_meta.json"), "r") as f:
                data = json.load(f)
            # Zeiträume ordnen
            temporal_extents = sorted(data["temporal_extents"], key=lambda x: x[0])
            ret = []
            # von vorne nach hinten durch die Zeiträume loopen, dabei start_calc (temporär zum berechnen) immer aufs Ende weitersetzen
            start_calc = start
            for _start, _end in temporal_extents:
                #print(f"comparing [{start_calc},{end}] to [{_start},{_end}]")
                if self._date_compare(_start, start_calc) < 0:
                    # wenn das neue Interval vor dem vorhandenen Interval startet
                    #print(f"{start_calc} is before {_start}")
                    if self._date_compare(_start, end) < 0:
                        # und wenn das neue Interval auch vor dem vorhandenen endet kann es ganz eingefügt werden
                        #print(f"{end} is also before {_start}")
                        ret += [[start_calc, end]]
                        return ret
                    else:
                        # und wenn das neue Interval nach dem Start des vorhandenen endet kann es bis zum Start des vorhandenen eingefügt werden
                        # start_calc wird auf das Ende des vorhandenen Intervals gesetzt, bzw. einen Tag danach
                        #print(f"{end} is at least {_start}")
                        ret += [[start_calc, self._date_add(_start, -1)]]
                        start_calc = self._date_add(_end, 1)
                elif self._date_compare(_end, start_calc) <= 0:
                    # wenn das neue Interval innerhalb eines Vorhandenen startet
                    #print(f"{start_calc} is maximal {_end}")
                    if self._date_compare(_end, end) <= 0:
                        # und wenn es auch innerhalb eines Vorhandenen endet kann es ignoriert werden
                        #print(f"{end} is maximal {_end}")
                        return ret
                    else:
                        # und wenn es danach endet kann die Berechnung mit dem nächsten Interval weitergehen
                        start_calc = self._date_add(_end, 1)
            ret += [[start_calc, end]]
            return ret
        except:
            return [[start, end]]

    # returns the difference in days between two dates
    # <0 if date1 > date2
    # >0 if date1 < date2
    # =0 if date1 = date2
    def _date_compare(self, date1, date2):
        _date1 = datetime.strptime(date1, '%Y-%m-%d')
        _date2 = datetime.strptime(date2, '%Y-%m-%d')
        return (_date2 - _date1).days

    def _concat_temporal_extents(self, temporal_extents1, temporal_extents2):
        '''angrenzende Zeiträume verbinden'''
        temporal_extents = sorted(temporal_extents1 + temporal_extents2, key=lambda x: x[0])
        i = 0
        while i < len(temporal_extents) - 1:
            _current = temporal_extents[i]
            _next = temporal_extents[i+1]

            if self._date_compare(_current[1], _next[0]) <= 1 or self._date_compare(_current[0], _next[0]) == 0:
                _current[1] = max(_current[1], _next[1])
                temporal_extents.pop(i+1)
            else:
                i += 1
        return temporal_extents
    
    def _update_meta(self, new_json, west, east, south, north, start, end):
        '''metadaten aktualisieren'''
        subfolder = self.subfolder_name(new_json["coords"]["x"]["data"], new_json["coords"]["y"]["data"], new_json["coords"]["spatial_ref"]["attrs"]["crs_wkt"])
        # zuerst die allgemeinen, also den Unterordner
        try:
            with open(os.path.join(self._output_dir, "_meta.json"), "r") as f:
                data = json.load(f)
        except:
            data = {}
        if f"{west:.8f},{south:.8f}-{east:.8f},{north:.8f}" not in data.keys():
            data[f"{west:.8f},{south:.8f}-{east:.8f},{north:.8f}"] = {}
        data[f"{west:.8f},{south:.8f}-{east:.8f},{north:.8f}"]["subfolder"] = subfolder
        # dann die heruntergeladenen Zeiträume
        with open(os.path.join(self._output_dir, "_meta.json"), "w") as f:
            json.dump(data, f, indent=4)
        try:
            with open(os.path.join(self._output_dir, subfolder, "_meta.json"), "r") as f:
                data = json.load(f)
        except:
            data = {}
            data["temporal_extents"] = []
        data["temporal_extents"] = self._concat_temporal_extents(data["temporal_extents"], [[start, end]])
        with open(os.path.join(self._output_dir, subfolder, "_meta.json"), "w") as f:
            json.dump(data, f, indent=4)

    def get_cloudless_dates(self, west, east, south, north, start, end):
        # https://developers.arcgis.com/python-2-3/samples/cloud-detector-part1-cloudless-sentinel-and-unsupervised/
        subfolder = self._get_subfolder(west, east, south, north)
        available_dates = self.get_available_dates(west, east, south, north, start, end)
        cloud_detector = S2PixelCloudDetector(
            threshold=0.5, # cloud probability threshold value. All pixels with cloud probability above threshold value are masked as cloudy pixels
            average_over=4, # Size of the disk in pixels for performing convolution (averaging probability over pixels)
            dilation_size=2, # Size of the disk in pixels for performing dilation
            all_bands=False # Flag specifying that input images will consists of all 13 Sentinel-2 bands. It has to be set to True if we would download all bands. If you define a layer that would return only 10 bands, then this parameter should be set to False
        )
        # Banddaten für alle verfügbaren Tage laden
        data = {}
        for date in available_dates:
            data[date] = {}
            for band in self.bands['L1C'].keys():
                data[date][band] = self.array_from_tiff(subfolder, date, 'L1C', band, 'float')
        ret = []
        # den s2cloudless Algorithmus ausführen
        for date, band_data in data.items():
            arr_list = [band_data[band] for band in ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']]
            arr = np.stack(arr_list, axis=-1)
            arr = np.expand_dims(arr, axis=0) # only single image for s2cloudless
            cloud_mask = cloud_detector.get_cloud_masks(arr)
            if np.mean(cloud_mask == 0) > 0.95:
                ret += [date]
        return ret

    def get_available_dates(self, west, east, south, north, start, end):
        '''alles heruntergeladenen Daten zurückgeben'''
        subfolder = self._get_subfolder(west, east, south, north)
        tif_files = glob.glob(os.path.join(self._output_dir, subfolder, '*.tif'))
        ret = []
        for file in tif_files:
            match = re.search(r'\d{4}-\d+-\d+', os.path.basename(file))
            if match:
                date = match.group()
                if self._date_compare(start, date) >= 0 and self._date_compare(date, end) >= 0:
                    ret += [date]
        return sorted(list(set(ret)))

    def get_index(self, date, index, west, east, south, north, save_result_as_tiff=False):
        '''Indices berechnen, RGB ist zwar soweit kein Index, aber funktioniert ähnlich'''
        index = index.lower()
        subfolder = self._get_subfolder(west, east, south, north)
        geoattrs = self.get_geoattrs(west, east, south, north)
        if index == 'ndvi':
            b04_data = np.clip(self.array_from_tiff(subfolder, date, 'L2A', 'B04', 'uint16'), 1, 65535).astype(np.float32)
            b08_data = np.clip(self.array_from_tiff(subfolder, date, 'L2A', 'B08', 'uint16'), 1, 65535).astype(np.float32)
            ndvi = np.clip(((b08_data - b04_data) / (b08_data + b04_data)) * 65535, 0, 65535).astype(np.uint16)
            if save_result_as_tiff:
                self.uint16_array_to_tiff(ndvi, subfolder, date, 'L2A', 'NDVI', ndvi.shape[0], ndvi.shape[1], geoattrs['crs'], geoattrs['transform'])
            return ndvi
        elif index == 'rgb':
            # see copernicus browser
            maxR = 3.0
            midR = 0.13
            sat = 1.2
            gamma = 1.8
            gOff = 0.01
            gOffPow = gOff ** gamma
            gOffRange = ((1 + gOff) ** gamma) - gOffPow
            def adjGamma(b):
                return (((b + gOff) ** gamma) - gOffPow) / gOffRange
            def clip(s):
                return max(0, min(1, s))
            def adj(a, tx, ty, maxC):
                ar = clip(a / maxC)
                return ar * (ar * (tx / maxC + ty - 1) - ty) / (ar * (2 * tx / maxC - 1) - tx / maxC)
            def sAdj(a):
                return adjGamma(adj(a, midR, 1, maxR))
            def satEnh(r, g, b):
                avgS = (r + g + b) / 3.0 * (1 - sat)
                return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)]
            def sRGB(c):
                if c <= 0.0031308:
                    return 12.92 * c
                else:
                    return 1.055 * (c ** (1 / 2.4)) - 0.055

            red_data = self.array_from_tiff(subfolder, date, 'L2A', 'B04', 'float')
            green_data = self.array_from_tiff(subfolder, date, 'L2A', 'B03', 'float')
            blue_data = self.array_from_tiff(subfolder, date, 'L2A', 'B02', 'float')
            for r, g, b in np.nditer([red_data, green_data, blue_data], op_flags=['readwrite']):
                rgbLin = satEnh(sAdj(r), sAdj(g), sAdj(b))
                r[...] = sRGB(rgbLin[0])
                g[...] = sRGB(rgbLin[1])
                b[...] = sRGB(rgbLin[2])
            rgb_data = np.stack([red_data, green_data, blue_data])
            rgb_data = np.clip(rgb_data * 65535, 0, 65535).astype(np.uint16)
            if save_result_as_tiff:
                self.uint16_array_to_tiff(rgb_data, subfolder, date, 'L2A', 'RGB', rgb_data.shape[1], rgb_data.shape[2], geoattrs['crs'], geoattrs['transform'], photometric='rgb')
            return rgb_data
        raise Exception(f"index {index} is not yet implemented")

    def get_geoattrs(self, west, east, south, north):
        '''Geodaten aus einer Geotiff Datei laden'''
        subfolder = self._get_subfolder(west, east, south, north)
        tiff_files = glob.glob(os.path.join(self._output_dir, subfolder, '*.tif'))
        if not tiff_files:
            raise Exception(f"unable to find *.tif in {subfolder}")
        if use_rasterio:
            with rasterio.open(tiff_files[0]) as src:
                return {
                    "crs": src.crs,
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "shape": src.shape
                }
        else:
            ds = gdal.Open(tiff_files[0])
            if ds is None:
                raise Exception(f"cannot open {tiff_files[0]}")
            proj = osr.SpatialReference(wkt=ds.GetProjection())
            crs = proj.ExportToProj4()
            transform = ds.GetGeoTransform()
            width = ds.RasterXSize
            height = ds.RasterYSize
            minx, maxy = transform[0], transform[3]
            maxx = minx + width * transform[1] + height * transform[2]
            miny = maxy + width * transform[4] + height * transform[5]
            return {
                "crs": crs,
                "transform": transform,
                "bounds": (minx, miny, maxx, maxy),
                "shape": (height, width)
            }
