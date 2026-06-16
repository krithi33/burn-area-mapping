"""
visualise.py
============
All mapping and plotting utilities:
  - Folium interactive map (burn severity + RF probability overlay)
  - dNBR histogram with threshold annotations
  - Feature importance bar chart
  - Method comparison figure (threshold vs RF side-by-side)
  - Confusion matrix heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from folium import plugins
import rasterio
from rasterio.warp import transform_bounds
from pathlib import Path

from .indices import DNBR_THRESHOLDS, SEVERITY_COLORS


# ── Colour utilities ───────────────────────────────────────────────────────

SEVERITY_CMAP = mcolors.ListedColormap(list(SEVERITY_COLORS.values()))
SEVERITY_BOUNDS = [-np.inf, -0.25, 0.10, 0.27, 0.44, 0.66, np.inf]
SEVERITY_NORM = mcolors.BoundaryNorm(
    [-0.5, -0.25, 0.10, 0.27, 0.44, 0.66, 1.5], SEVERITY_CMAP.N
)


def _build_legend_patches(label_map: dict) -> list:
    return [
        mpatches.Patch(color=color, label=label)
        for label, color in label_map.items()
    ]


# ── dNBR Histogram ─────────────────────────────────────────────────────────

def plot_dnbr_distribution(
    dNBR: np.ndarray,
    out_path: str | Path | None = None
) -> plt.Figure:
    """
    Distribution of dNBR values with USGS threshold lines annotated.
    This plot goes in the README — it shows the burn signal is real.
    """
    flat = dNBR.ravel()
    flat = flat[np.isfinite(flat)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(flat, bins=200, color="#444", alpha=0.75, range=(-0.8, 1.2))

    threshold_labels = {
        -0.25: "Regrowth",
        0.10:  "Low",
        0.27:  "Mod-Low",
        0.44:  "Mod-High",
        0.66:  "High"
    }
    for val, label in threshold_labels.items():
        ax.axvline(val, color="#d73027", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.text(val + 0.01, ax.get_ylim()[1] * 0.85, label,
                fontsize=8, color="#d73027", rotation=90, va="top")

    ax.set_xlabel("dNBR", fontsize=12)
    ax.set_ylabel("Pixel count", fontsize=12)
    ax.set_title("dNBR Distribution — Dixie Fire 2021\n(USGS burn severity thresholds annotated)",
                 fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


# ── Severity Map ───────────────────────────────────────────────────────────

def plot_severity_map(
    dNBR: np.ndarray,
    title: str = "Burn Severity — Dixie Fire 2021 (Threshold)",
    out_path: str | Path | None = None
) -> plt.Figure:
    """Static matplotlib map of dNBR-based burn severity classes."""
    fig, ax = plt.subplots(figsize=(12, 9))

    im = ax.imshow(
        dNBR,
        cmap=SEVERITY_CMAP,
        norm=SEVERITY_NORM,
        interpolation="nearest"
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.axis("off")

    patches = _build_legend_patches(SEVERITY_COLORS)
    ax.legend(handles=patches, loc="lower left", framealpha=0.85,
              title="Burn Severity", title_fontsize=9, fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


# ── Method Comparison Figure ───────────────────────────────────────────────

def plot_method_comparison(
    threshold_map: np.ndarray,
    rf_map: np.ndarray,
    proba_map: np.ndarray,
    out_path: str | Path | None = None
) -> plt.Figure:
    """
    Three-panel figure:
      Left   : Threshold binary map
      Centre : RF binary map
      Right  : RF burn probability (continuous)

    This is the key visual for your README and any presentations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    panels = [
        (threshold_map, "Threshold (dNBR ≥ 0.27)", "RdYlGn_r"),
        (rf_map,        "Random Forest (binary)",   "RdYlGn_r"),
        (proba_map,     "RF Burn Probability",      "YlOrRd"),
    ]

    for ax, (arr, title, cmap) in zip(axes, panels):
        vmin, vmax = (0, 1) if arr.max() <= 1 else (arr.min(), arr.max())
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=12, pad=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)

    fig.suptitle("Burn Area Detection — Dixie Fire 2021\nMethod Comparison",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


# ── Feature Importance Chart ───────────────────────────────────────────────

def plot_feature_importance(
    importance_df,
    top_n: int = 15,
    out_path: str | Path | None = None
) -> plt.Figure:
    """Horizontal bar chart of RF feature importances (top N features)."""
    df = importance_df.head(top_n)
    colors = ["#d73027" if "dNBR" in f or "NBR" in f else "#4575b4"
              for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title(f"RF Feature Importance (Top {top_n})", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#d73027", label="Burn-related (NBR family)"),
        Patch(color="#4575b4", label="Vegetation / reflectance"),
    ]
    ax.legend(handles=legend, fontsize=9)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


# ── Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list | np.ndarray,
    title: str = "Confusion Matrix",
    out_path: str | Path | None = None
) -> plt.Figure:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Unburned", "Burned"],
        yticklabels=["Unburned", "Burned"],
        ax=ax
    )
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ── Folium Interactive Map ─────────────────────────────────────────────────

def build_folium_map(
    tif_path: str | Path,
    threshold_map: np.ndarray,
    rf_map: np.ndarray,
    proba_map: np.ndarray,
    out_path: str | Path = "outputs/maps/burn_map_interactive.html",
    perimeter_path: str | Path | None = None
) -> folium.Map:
    """
    Build a layered Folium map with:
      - Basemap (OpenStreetMap + Esri Satellite toggle)
      - Threshold burn mask (red overlay)
      - RF burn mask (orange overlay)
      - RF probability heatmap (continuous)
      - Layer control
      - Legend
      - Fullscreen button

    All layers are raster overlays from in-memory numpy arrays.
    We reproject bounds to WGS84 for Folium (which uses Leaflet/lat-lon).
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable, YlOrRd
    import io
    import base64
    from PIL import Image as PILImage

    # Get map bounds in WGS84 from the raster
    with rasterio.open(tif_path) as src:
        bounds_utm = src.bounds
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *bounds_utm)

    west, south, east, north = bounds_wgs84
    center = [(south + north) / 2, (west + east) / 2]

    m = folium.Map(
        location=center,
        zoom_start=9,
        tiles="OpenStreetMap"
    )

    # Esri satellite basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False
    ).add_to(m)

    def array_to_image_overlay(arr, colormap, name, opacity=0.6):
        """Convert numpy array → PNG → base64 → ImageOverlay on Folium map."""
        norm = Normalize(vmin=arr.min(), vmax=arr.max())
        sm = ScalarMappable(norm=norm, cmap=colormap)
        rgba = sm.to_rgba(arr, bytes=True)
        # Mask: where arr == 0, set alpha to 0 (transparent)
        if arr.max() <= 1:
            rgba[arr == 0, 3] = 0
        img = PILImage.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        url = f"data:image/png;base64,{b64}"
        return folium.raster_layers.ImageOverlay(
            image=url,
            bounds=[[south, west], [north, east]],
            opacity=opacity,
            name=name,
            interactive=True,
            cross_origin=False,
            zindex=1
        )

    array_to_image_overlay(threshold_map, "Reds",   "Threshold Burn Mask", 0.55).add_to(m)
    array_to_image_overlay(rf_map,        "Oranges", "RF Burn Mask",        0.55).add_to(m)
    array_to_image_overlay(proba_map,     YlOrRd,    "RF Burn Probability", 0.65).add_to(m)

    # Layer control
    # Official CAL FIRE perimeter boundary (optional)
    if perimeter_path and Path(perimeter_path).exists():
        import geopandas as gpd
        gdf = gpd.read_file(perimeter_path).to_crs("EPSG:4326")
        folium.GeoJson(
            gdf.__geo_interface__,
            name="CAL FIRE Official Perimeter",
            style_function=lambda x: {
                "color": "#ffffff",
                "weight": 2.5,
                "fillOpacity": 0,
                "dashArray": "6 4"
            },
            tooltip="Official Dixie Fire Perimeter (CAL FIRE)"
        ).add_to(m)
        print("  Perimeter layer added to map")

    folium.LayerControl(collapsed=False).add_to(m)

    # Fullscreen
    plugins.Fullscreen(position="topright").add_to(m)

    # Mini-map
    plugins.MiniMap().add_to(m)

    # Legend HTML
    legend_html = """
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 1000;
        background: rgba(255,255,255,0.9); padding: 12px 16px;
        border-radius: 8px; border: 1px solid #ccc;
        font-family: Arial, sans-serif; font-size: 13px;
    ">
        <b>Burn Severity (Threshold)</b><br>
        <span style="background:#1a9641;padding:2px 10px;">&nbsp;</span> Enhanced Regrowth<br>
        <span style="background:#a6d96a;padding:2px 10px;">&nbsp;</span> Unburned<br>
        <span style="background:#ffffbf;padding:2px 10px;">&nbsp;</span> Low Severity<br>
        <span style="background:#fdae61;padding:2px 10px;">&nbsp;</span> Moderate-Low<br>
        <span style="background:#f46d43;padding:2px 10px;">&nbsp;</span> Moderate-High<br>
        <span style="background:#d73027;padding:2px 10px;">&nbsp;</span> High Severity<br>
        <hr style="margin:6px 0">
        <span style="color:#666;font-size:11px;">Source: Sentinel-2 SR | 20m resolution</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title
    title_html = """
    <div style="
        position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: rgba(255,255,255,0.92);
        padding: 8px 20px; border-radius: 6px; border: 1px solid #bbb;
        font-family: Arial, sans-serif; font-size: 15px; font-weight: bold;
    ">
        🔥 Dixie Fire 2021 — Burn Area Detection (Sentinel-2)
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"Interactive map saved → {out_path}")
    return m
