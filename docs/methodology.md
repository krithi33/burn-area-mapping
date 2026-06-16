# Technical Methodology

## Sensor & Data Selection

### Why Sentinel-2 Surface Reflectance?

Sentinel-2 Level-2A (Surface Reflectance) corrects for atmospheric scattering and absorption using the Sen2Cor processor. This is critical for spectral index computation because:

- **TOA (Top of Atmosphere) reflectance** includes aerosol and water vapour effects that vary between acquisition dates
- A cloud of wildfire smoke present in the pre-fire image would artificially suppress NIR reflectance, inflating dNBR even without actual fire damage
- SR removes these effects, making pre/post comparisons physically meaningful

### Band Selection for NBR

Standard NBR uses the SWIR-2 band (Sentinel-2 B12, 2190nm) rather than SWIR-1 (B11, 1610nm). Empirical studies show B12 provides stronger contrast between burned and unburned areas because:

1. Char and ash have high reflectance specifically at ~2.2µm
2. Green vegetation absorbs strongly at 2.2µm (cellulose absorption feature)
3. B11 at 1.6µm is less sensitive to moisture content differences between burned/unburned

## Composite Strategy

A **median composite** over a date range is used rather than a single image because:

- Individual Sentinel-2 tiles have residual cloud/shadow artifacts even after masking
- Median suppresses these outliers: a cloud reflects near-zero NIR and very high SWIR, pulling NBR sharply negative. The median across 10+ images ignores these extremes.
- Mean would be pulled toward cloud-contaminated values

## Threshold Parameters

The USGS dNBR thresholds (Key & Benson 2006) were developed through field campaigns measuring basal area mortality and canopy cover loss. The thresholds are:

| Value | Ecological meaning |
|---|---|
| > 0.66 | >75% canopy mortality, exposed mineral soil |
| 0.44–0.66 | 50–75% canopy mortality |
| 0.27–0.44 | 25–50% canopy mortality |
| 0.10–0.27 | <25% canopy mortality, minor scorch |
| -0.25–0.10 | Negligible change |
| < -0.25 | Post-fire green-up (grasses, shrubs resprouting) |

## Random Forest Design Decisions

### Why 16 Features (Not Just dNBR)?

Using only dNBR would essentially replicate the threshold classifier. The RF adds value by learning non-linear combinations:

- **Pre-fire NDVI + post-fire NDVI together** distinguish naturally low-NDVI areas (desert margins, roads) from fire-caused NDVI drops
- **BAI (Burn Area Index)** is specifically tuned to the char spectral signature; it amplifies the signal in high-severity areas where NBR may saturate
- **NDWI (canopy moisture)** pre-fire separates wet forest from dry chaparral, which affects how severely each burns

### Class Imbalance Handling

Burned pixels are a minority class (typically 15-30% of any regional AOI). Without correction, RF optimises overall accuracy by predicting "unburned" everywhere, achieving 80%+ accuracy while detecting nothing.

Solutions applied:
1. `class_weight='balanced'` — scales class weights inversely proportional to frequency
2. Stratified sampling in GEE — guarantees equal burned/unburned training samples
3. Evaluation with F1 and Average Precision (not accuracy) — metrics robust to class imbalance

### Spatial Autocorrelation Warning

The training/test split used is random (not spatial). Nearby pixels share spectral values due to sensor blur and atmospheric continuity. This means test set pixels are likely spatially adjacent to training pixels → optimistic performance estimates.

A production implementation should use **spatial block cross-validation** where spatially contiguous blocks are held out entirely. This is noted as a limitation in the README.

## RdNBR: Relativised dNBR

Miller & Thode (2007) showed that standard dNBR systematically underestimates severity in areas with low pre-fire vegetation density. A shrubland with pre-fire NBR of 0.2 that burns completely (post-fire NBR = -0.1) gives dNBR = 0.3 (Moderate-Low). A dense forest with pre-fire NBR = 0.7 that burns at the same severity (post-fire NBR = 0.4) gives dNBR = 0.3 (also Moderate-Low). But field measurements show the shrubland experienced *higher proportional* canopy loss.

RdNBR = dNBR / √|pre_NBR| normalises by the pre-fire vegetation signal, producing comparable severity estimates regardless of starting biomass.

## Validation Approach

Primary validation in this project uses **dNBR-thresholded proxy labels** for RF training. This creates a circular dependency (threshold labels used to evaluate a method that's supposed to improve on threshold). 

The correct validation approach — planned as a future enhancement — uses the official **CAL FIRE FRAP perimeter polygon** as binary ground truth:
- Inside perimeter: labeled burned
- Outside perimeter: labeled unburned

The perimeter is downloadable from: https://www.fire.ca.gov/incidents/2021/7/13/dixie-fire/
