"""
gee_export.py
=============
Handles all Google Earth Engine interactions:
  - AOI definition
  - Sentinel-2 SR collection filtering
  - Cloud masking
  - Spectral index computation (NBR, NDVI, NDWI, BAI)
  - Export to Google Drive (GeoTIFF + training sample CSV)

Run this first. Everything downstream reads from the exported GeoTIFFs.
"""

import ee
import geemap


# ── Constants ──────────────────────────────────────────────────────────────

# 2021 Dixie Fire — largest single wildfire in California history
FIRE_NAME = "Dixie Fire 2021"
AOI_BOUNDS = dict(west=-121.5, south=39.7, east=-120.3, north=40.6)

PRE_FIRE_START  = "2021-05-01"
PRE_FIRE_END    = "2021-07-10"   # Fire ignition: July 13 2021
POST_FIRE_START = "2021-10-26"   # Containment:   Oct 25 2021
POST_FIRE_END   = "2021-11-30"

CLOUD_THRESHOLD = 20             # max % cloud cover per image
EXPORT_SCALE    = 20             # metres — S2 SWIR native resolution
EXPORT_CRS      = "EPSG:32610"   # UTM Zone 10N (Northern California)
EXPORT_FOLDER   = "BurnMapping_Dixie2021"
SAMPLE_POINTS   = 5000           # per class (burned / unburned)
RANDOM_SEED     = 42


# ── GEE Initialisation ─────────────────────────────────────────────────────

def init_gee(project_id: str) -> None:
    """Authenticate and initialise GEE. Run ee.Authenticate() once before."""
    ee.Initialize(project=project_id)
    print(f"GEE initialised → project: {project_id}")


# ── Cloud Masking ──────────────────────────────────────────────────────────

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Mask clouds and cirrus using Sentinel-2 QA60 band.
    Also scales reflectance from [0, 10000] → [0, 1].

    Bit 10 = opaque clouds | Bit 11 = cirrus clouds
    Pixels where either flag is set are masked out.
    """
    qa = image.select("QA60")
    cloud_mask  = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
    return (
        image
        .updateMask(cloud_mask.And(cirrus_mask))
        .divide(10000)
        .copyProperties(image, ["system:time_start"])
    )


# ── Spectral Indices ───────────────────────────────────────────────────────

def add_indices(image: ee.Image) -> ee.Image:
    """
    Compute and append spectral indices used in burn analysis.

    NBR  = (B8 - B12) / (B8 + B12)   — core burn signal
    NDVI = (B8 - B4)  / (B8 + B4)    — vegetation greenness
    NDWI = (B3 - B8)  / (B3 + B8)    — canopy moisture
    BAI  = 1 / ((0.1-RED)² + (0.06-NIR)²) — char detection

    Using B12 (2190nm) not B11 (1610nm) for NBR — USGS standard for S2.
    """
    nbr  = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    bai  = image.expression(
        "1.0 / ((0.1 - RED)**2 + (0.06 - NIR)**2)",
        {"RED": image.select("B4"), "NIR": image.select("B8")}
    ).rename("BAI").toFloat()          # expression() returns Float64 — cast to Float32
    return image.addBands([nbr, ndvi, ndwi, bai])


# ── Image Collection ───────────────────────────────────────────────────────

def get_composite(
    aoi: ee.Geometry,
    start: str,
    end: str,
    label: str
) -> ee.Image:
    """
    Build a cloud-free median composite from S2 SR over a date range.
    Returns image with spectral indices appended.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESHOLD))
        .map(mask_s2_clouds)
        .map(add_indices)
    )
    count = collection.size().getInfo()
    print(f"  {label}: {count} images → median composite")
    return collection.median().clip(aoi)


# ── Analysis Stack ─────────────────────────────────────────────────────────

def build_analysis_stack(
    pre: ee.Image,
    post: ee.Image
) -> tuple[ee.Image, ee.Image, ee.Image]:
    """
    Combine pre/post composites into a labelled analysis stack.
    Returns (analysis_stack, dNBR, RdNBR).

    dNBR  = pre_NBR - post_NBR
    RdNBR = dNBR / sqrt(|pre_NBR|)  [Miller & Thode 2007]
    """
    bands = ["NBR", "NDVI", "NDWI", "BAI", "B4", "B8", "B11", "B12"]

    pre_renamed  = pre.select(bands).rename([f"pre_{b}"  for b in bands])
    post_renamed = post.select(bands).rename([f"post_{b}" for b in bands])

    dNBR  = pre.select("NBR").subtract(post.select("NBR")).rename("dNBR")
    rdNBR = dNBR.divide(
        pre.select("NBR").abs().sqrt()
    ).rename("RdNBR")

    stack = pre_renamed.addBands(post_renamed).addBands([dNBR, rdNBR])
    return stack, dNBR, rdNBR


# ── Exports ────────────────────────────────────────────────────────────────

def export_raster(image: ee.Image, name: str, aoi: ee.Geometry) -> ee.batch.Task:
    """Export an ee.Image to Google Drive as GeoTIFF at 20m / UTM Zone 10N."""
    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),         # guarantee all bands are Float32 before export
        description=name,
        folder=EXPORT_FOLDER,
        region=aoi,
        scale=EXPORT_SCALE,
        crs=EXPORT_CRS,
        maxPixels=1_000_000_000,
        fileFormat="GeoTIFF"
    )
    task.start()
    print(f"  Export queued: {name}.tif → Drive/{EXPORT_FOLDER}/")
    return task


def export_training_samples(
    stack: ee.Image,
    dNBR: ee.Image,
    aoi: ee.Geometry
) -> ee.batch.Task:
    """
    Stratified sample: SAMPLE_POINTS burned + SAMPLE_POINTS unburned pixels.
    Label proxy: dNBR > 0.27 = burned (moderate+ severity threshold, USGS).
    """
    labeled = stack.addBands(dNBR.gt(0.27).rename("burned_label"))
    sample  = labeled.stratifiedSample(
        numPoints=SAMPLE_POINTS,
        classBand="burned_label",
        region=aoi,
        scale=EXPORT_SCALE,
        seed=RANDOM_SEED,
        geometries=True
    )
    task = ee.batch.Export.table.toDrive(
        collection=sample,
        description="training_samples_dixie2021",
        folder=EXPORT_FOLDER,
        fileFormat="CSV"
    )
    task.start()
    print(f"  Training samples queued: {SAMPLE_POINTS*2} labeled pixels → CSV")
    return task


# ── Main ───────────────────────────────────────────────────────────────────

def run_export(project_id: str) -> None:
    """Full pipeline: init → pull imagery → compute indices → export."""
    init_gee(project_id)

    aoi = ee.Geometry.BBox(**AOI_BOUNDS)
    print(f"\nTarget: {FIRE_NAME}")
    print(f"AOI: {AOI_BOUNDS}")

    print("\n[1/3] Pulling imagery ...")
    pre  = get_composite(aoi, PRE_FIRE_START,  PRE_FIRE_END,  "Pre-fire")
    post = get_composite(aoi, POST_FIRE_START, POST_FIRE_END, "Post-fire")

    print("\n[2/3] Building analysis stack ...")
    stack, dNBR, rdNBR = build_analysis_stack(pre, post)

    print("\n[3/3] Queuing exports ...")
    export_raster(pre,   "pre_fire_S2_dixie2021",      aoi)
    export_raster(post,  "post_fire_S2_dixie2021",     aoi)
    export_raster(stack, "analysis_stack_dixie2021",   aoi)
    export_training_samples(stack, dNBR, aoi)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  All exports queued. Monitor at:                         ║
║  https://code.earthengine.google.com/tasks               ║
║                                                          ║
║  Expected outputs in Drive/{EXPORT_FOLDER}/     ║
║    pre_fire_S2_dixie2021.tif                             ║
║    post_fire_S2_dixie2021.tif                            ║
║    analysis_stack_dixie2021.tif                          ║
║    training_samples_dixie2021.csv                        ║
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "your-gee-project-id"
    run_export(project)
