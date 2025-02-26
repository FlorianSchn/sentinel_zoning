from sentinel_zoning import Sentinel2, Zoning
import json
import pyperclip

def main():
    job = {}
    # die jobbeschreibung annehmen
    while job == {}:
        try:
            job = json.loads(input('Zonierungsauftrag einfügen: '))
        except KeyboardInterrupt:
            return
        except:
            print("Zonierungsauftrag konnte nicht gelesen werden")
    #sentinel = Sentinel2('openeo.dataspace.copernicus.eu', 'sentinel2_data', job['force_auth'])
    sentinel = Sentinel2('openeo.dataspace.copernicus.eu', 'sentinel2_data', False)
    zoning = Zoning(sentinel)
    west = job['west']
    east = job['east']
    south = job['south']
    north = job['north']
    wkt = job['wkt']
    num_zones = int(job['zones'])
    # Zeiträume bzw. Vegetationsperioden laden oder neu eingeben
    timeperiods = sentinel.get_cached_timezones(west, east, south, north)
    if timeperiods != []:
        print("")
        for timeperiod in timeperiods:
            print(f"{timeperiod[0]} bis {timeperiod[1]}")
        use_cache = input("sollen diese gespeicherten Zeiträume verwendet werden? [y]/n: ").lower().strip()
        if use_cache == 'n':
            timeperiods = []
    # Zeiträume neu eingeben
    if timeperiods == []:
        while True:
            period = input("Zeitraum Start und Ende eingeben im Format yyyy-mm-dd yyyy-mm-dd oder Enter: ")
            # TODO Eingabe auf Fehler überprüfen
            if not period:
                break
            timeperiods += [period.split()]
    cloudless_dates = []
    for start, end in timeperiods:
        # sicherstellen, dass die entsprechenden Satellitendaten lokal verfügbar sind
        sentinel.provide(west, east, south, north, start, end)
        # die Daten ohne Wolken berechnen
        cloudless_dates += sentinel.get_cloudless_dates(west, east, south, north, start, end)
    # cachen der Zeiträume
    sentinel.set_cached_timezones(west, east, south, north, timeperiods)
    if cloudless_dates:
        zones = zoning.zone(cloudless_dates, west, east, south, north, num_zones, wkt)
        print("")
        print(json.dumps(zones))
        print("")
        pyperclip.copy(json.dumps(zones))
        print("Ergebnis wurde in Zwischenablage kopiert")
    else:
        print("keine Daten ohne Wolken gefunden")

if __name__ == '__main__':
    main()
