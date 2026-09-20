# First Run — Download Weather Data

Before running the simulation, you need the PVGIS TMY (Typical Meteorological Year) weather data files. These are hourly climate files for each test site.

## Download Links

Use these URLs to download the TMY data directly from the PVGIS API (European Commission JRC):

### Ingolstadt, Germany (Primary Site)
```
https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat=48.76&lon=11.42&outputformat=csv
```
Save as: `data/tmy_ingolstadt.csv`

### Ghent, Belgium
```
https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat=51.05&lon=3.72&outputformat=csv
```
Save as: `data/tmy_belgium.csv`

### Vidsel, Sweden
```
https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat=59.33&lon=18.07&outputformat=csv
```
Save as: `data/tmy_sweden.csv`

## Alternative: Download via Browser

1. Go to [https://re.jrc.ec.europa.eu/pvg_tools/en/](https://re.jrc.ec.europa.eu/pvg_tools/en/)
2. Select "TMY data" tab
3. Enter latitude/longitude for your site
4. Click "Download"
5. Save the CSV file as `data/tmy_ingolstadt.csv` (or the appropriate name)

## Alternative: Download via Python

```python
import urllib.request

urls = {
    "data/tmy_ingolstadt.csv": "https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat=48.76&lon=11.42&outputformat=csv",
    "data/tmy_belgium.csv": "https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat=51.05&lon=3.72&outputformat=csv",
    "data/tmy_sweden.csv": "https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat=59.33&lon=18.07&outputformat=csv",
}

for filepath, url in urls.items():
    print(f"Downloading {filepath}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"  ✅ Saved to {filepath}")
```

## Verify Download

After downloading, check the files exist:
```bash
ls data/
# Should show: tmy_ingolstadt.csv, tmy_belgium.csv, tmy_sweden.csv
```

Each file should be ~200–500 KB with 8,760+ rows of hourly data.

## Then Run

```bash
python src/main.py
```

See [INSTALLATION.md](INSTALLATION.md) for setup instructions if you haven't installed dependencies yet.
