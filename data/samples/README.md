# data/samples/

Place the training samples CSV here after downloading from Google Drive.

Expected file:
- `training_samples_dixie2021.csv`  — 10,000 labeled pixels (5,000 burned / 5,000 unburned)

Columns: pre_NBR, pre_NDVI, pre_NDWI, pre_BAI, pre_B4, pre_B8, pre_B11, pre_B12,
         post_NBR, post_NDVI, post_NDWI, post_BAI, post_B4, post_B8, post_B11, post_B12,
         dNBR, RdNBR, burned_label (0/1), longitude, latitude

Labels are dNBR proxy labels (dNBR > 0.27 = burned).
Refined with CAL FIRE perimeter polygons in the analysis notebook.
