# Installation
- Visual C++ Redistributable installieren
  (nötig für den s2cloudless-Algorithmus)
  
  https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
- Python installieren (funktioniert auch ohne admin-Rechte, nur keine Optionen anklicken, die admin-Rechte brauchen):
- 
  https://www.python.org/downloads/
  - Haken bei "Add Python to PATH" setzen
- Python virtual environment erstellen, dafür einen lokalen Ordner verwenden
  `cmd` oder `powershell` ausführen und folgendes eingeben, danach für die weiteren Befehle offen lassen:
  ```python -m venv Pfad\zum\virtuellen\environment```
  (z.B. python -m venv C:\Users\...\Desktop\sentinel_zoning_venv oder python -m venv sentinel_venv)
- In der Kommandozeile in das Python Environment navigieren:
  ```cd Pfad\zum\virtuellen\environment`
- Environment aktivieren
  ```.\Scripts\activate```
- in der Kommandozeile Pakete für das Environment installieren:
  ```pip install -U openeo rasterio pyproj s2cloudless mapclassify pyperclip```
- sentinel_zoning bereitstellen:
   - den Ordner `sentinel_zoning` im virtuellen Environment in den Ordner `Lib\site-packages` kopieren
   - die `main.py` und `start_venv.bat` in den Ordner des virtuellen Environments kopieren

# Ausführung
- im Ordner des virtuellen Environments für sentinel_zoning die `start_venv.bat` ausführen
- im Konsolenfenster das Programm starten:
  ```python main.py```
- Zonierungsauftrag einfügen
- ausführen (Enter)
- Anpassungen beantworten
- warten
