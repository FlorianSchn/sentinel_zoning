# Installation
1. Visual C++ Redistributable installieren
   (nicht getestet, da keine Admin-Rechte, das ist aber so wie es ausschaut für den s2cloudless-Algorithmus nötig)
   https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
2. Python installieren (funktioniert auch ohne admin-Rechte, nur keine Optionen anklicken, die admin-Rechte brauchen):
   https://www.python.org/downloads/
   - Haken bei "Add Python to PATH" setzen
3. Python virtual environment erstellen, dafür einen lokalen Ordner (nicht auf OneDrive) verwenden
   Kommandozeile 'cmd' oder 'powershell' ausführen und folgendes eingeben, danach für die weiteren Befehle offen lassen:
   python -m venv Pfad\zum\virtuellen\environment
   (z.B. python -m venv C:\Users\...\Desktop\sentinel_zoning_venv)
   (oder python -m venv sentinel_venv)
4. In der Kommandozeile in das Python Environment navigieren:
   cd Pfad\zum\virtuellen\environment
5. Environment aktivieren
   .\Scripts\activate
6. in der Kommandozeile Pakete für das Environment installieren:
   pip install -U openeo rasterio pyproj s2cloudless mapclassify pyperclip
7. sentinel_zoning bereitstellen:
   - den Ordner sentinel_zoning im virtuellen Environment in den Ordner Lib\site-packages kopieren
   - die main.py und start_venv.bat in den Ordner des virtuellen Environments kopieren

# Ausführung
1. im Ordner des virtuellen Environments für sentinel_zoning die start_venv.bat ausführen
2. im Konsolenfenster das Programm starten:
   python main.py
3. die Ausgabe (unter Messages) von ArcGIS in das Programm einfügen (Rechtsklick)
4. ausführen (Enter)
5. Anpassungen beantworten
6. warten
7. Ergebnis in das Feld in ArcGIS kopieren und das Tool nochmal ausführen
