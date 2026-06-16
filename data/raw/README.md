# data/raw/

Place GeoTIFFs exported from Google Earth Engine here after downloading from Google Drive.

Expected files:
- `pre_fire_S2_dixie2021.tif`   — Pre-fire Sentinel-2 composite (May–Jul 2021)
- `post_fire_S2_dixie2021.tif`  — Post-fire Sentinel-2 composite (Oct–Nov 2021)

These files are excluded from version control (see .gitignore) due to size (~200MB each).
Run `src/gee_export.py` to regenerate them.
