"""
Script 02: Terrain Visualizations (improved)
- 3D surface + zone markers
- Contour + hillshade map with building overlay and zone boundaries
- Shade/shadow analysis
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pyproj import Transformer
from shapely.geometry import Point

OUT.mkdir(parents=True, exist_ok=True)

# ── matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.facecolor": "white",
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def _t4326_to_utm(lat, lon):
    tr = Transformer.from_crs(4326, EPSG_LOCAL, always_xy=True)
    return tr.transform(lon, lat)   # returns easting, northing

def _north_arrow(ax, x=0.96, y=0.14):
    ax.annotate("N", xy=(x, y + 0.055), xytext=(x, y),
                xycoords="axes fraction", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=2))

def _add_scalebar(ax):
    ax.add_artist(ScaleBar(1, units="m", location="lower left",
                           font_properties={"size": 8},
                           box_alpha=0.7, frameon=True))

def _zone_circles(ax, xs, ys, utm=True, linewidth=2.0):
    """Draw zone boundary circles and labels on the map."""
    for zk, z in ZONES.items():
        if utm:
            cx, cy = _t4326_to_utm(z["lat"], z["lon"])
        else:
            cx, cy = z["lon"], z["lat"]
        r = z["radius_m"] if utm else z["radius_m"] / 111000
        circle = plt.Circle((cx, cy), r, fill=False,
                             edgecolor=z["color"], linewidth=linewidth,
                             linestyle="--", zorder=6)
        ax.add_patch(circle)
        ax.annotate(z["name"], (cx, cy + r * 1.12),
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=z["color"], zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

def _zone_legend():
    return [mpatches.Patch(facecolor=z["light"], edgecolor=z["color"],
                           linewidth=1.5, label=z["name"])
            for z in ZONES.values()]

def _load_dem_utm():
    with rasterio.open(PROC / "dem_utm17s.tif") as src:
        data = src.read(1).astype(float)
        data = np.where(data < -1000, np.nan, data)
        t = src.transform
        r, c = data.shape
        xs = np.array([t.c + t.a * j for j in range(c)])
        ys = np.array([t.f + t.e * i for i in range(r)])
    return data, xs, ys

def _hillshade(dem, azimuth=315, altitude=45):
    az  = np.radians(360 - azimuth)
    alt = np.radians(altitude)
    dy, dx = np.gradient(dem)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)

def _load_vec(path, layer=None, crs=None):
    try:
        gdf = gpd.read_file(path, layer=layer)
        return gdf.to_crs(crs) if crs else gdf
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — 3D Terrain
# ══════════════════════════════════════════════════════════════════════════════
def plot_terrain_3d():
    print("── 3D terrain plot …")
    dem, xs, ys = _load_dem_utm()
    step = max(1, dem.shape[0] // 100)
    dem_s, xs_s, ys_s = dem[::step, ::step], xs[::step], ys[::step]
    XX, YY = np.meshgrid(xs_s / 1000, ys_s / 1000)

    fig = plt.figure(figsize=(13, 9), facecolor="white")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f8f8f8")

    surf = ax.plot_surface(XX, YY, dem_s, cmap="terrain",
                           linewidth=0, antialiased=True, alpha=0.92,
                           vmin=np.nanmin(dem_s), vmax=np.nanmax(dem_s))

    # Zone markers on surface
    tr = Transformer.from_crs(4326, EPSG_LOCAL, always_xy=True)
    for zk, z in ZONES.items():
        ex, ey = tr.transform(z["lon"], z["lat"])
        # find nearest grid indices
        ci = np.argmin(np.abs(xs_s - ex))
        ri = np.argmin(np.abs(ys_s - ey))
        elev = dem_s[ri, ci] + 30
        ax.scatter([ex/1000], [ey/1000], [elev], c=z["color"],
                   s=80, zorder=10, depthshade=False)
        ax.text(ex/1000, ey/1000, elev + 15, z["name"],
                fontsize=7, color=z["color"], fontweight="bold",
                ha="center")

    cbar = fig.colorbar(surf, ax=ax, shrink=0.38, aspect=12, pad=0.08)
    cbar.set_label("Elevación (m s.n.m.)", fontsize=9)
    ax.set_xlabel("Este (km)", labelpad=8, fontsize=9)
    ax.set_ylabel("Norte (km)", labelpad=8, fontsize=9)
    ax.set_zlabel("Elevación (m)", labelpad=8, fontsize=9)
    ax.set_title("Yachay Tech — Modelo Digital de Elevación (3D)\nUrcuquí, Imbabura, Ecuador",
                 **TITLE_FONT)
    ax.view_init(elev=30, azim=-60)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, "01_terrain_3d")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Contour + Hillshade
# ══════════════════════════════════════════════════════════════════════════════
def plot_contours():
    print("── Contour + hillshade map …")
    dem, xs, ys = _load_dem_utm()
    hs = _hillshade(dem, azimuth=315, altitude=40)
    XX, YY = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(11, 11), facecolor="white")
    ax.set_facecolor("#e8f0f7")

    # Hillshade
    ax.imshow(hs, extent=[xs.min(), xs.max(), ys.min(), ys.max()],
              cmap="gray", vmin=0, vmax=1, alpha=0.55,
              origin="lower", aspect="equal")

    # Elevation colour
    cmap_terrain = plt.cm.get_cmap("terrain")
    elev_img = ax.imshow(dem, extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                          cmap=cmap_terrain, alpha=0.6, origin="lower", aspect="equal",
                          vmin=np.nanmin(dem), vmax=np.nanmax(dem))

    # Contours every 10 m
    step = max(5, int((np.nanmax(dem) - np.nanmin(dem)) / 30))
    levels = np.arange(np.floor(np.nanmin(dem) / step) * step,
                       np.nanmax(dem) + step, step)
    cs = ax.contour(XX, YY, dem, levels=levels, colors="#333", linewidths=0.35, alpha=0.6)
    # Label every 2nd level
    label_levels = levels[::2]
    ax.clabel(cs, levels=label_levels, inline=True, fontsize=6.5, fmt="%d m",
              colors="#333")

    # Buildings
    blds = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)
    if blds is not None and len(blds):
        blds.plot(ax=ax, facecolor="#fce8c8", edgecolor="#8b5e3c", linewidth=0.8,
                  alpha=0.85, zorder=3)

    # Roads
    roads = _load_vec(RAW / "roads.gpkg", "roads", EPSG_LOCAL)
    if roads is not None and len(roads):
        roads.plot(ax=ax, color="#555", linewidth=0.5, alpha=0.6, zorder=2)

    # Zone circles
    _zone_circles(ax, xs, ys, utm=True)

    # Colourbar
    cbar = fig.colorbar(elev_img, ax=ax, fraction=0.025, pad=0.02, shrink=0.8)
    cbar.set_label("Elevación (m s.n.m.)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    _add_scalebar(ax)
    _north_arrow(ax)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#fce8c8", edgecolor="#8b5e3c", label="Edificaciones"),
        mpatches.Patch(color="#555", label="Vías"),
    ] + _zone_legend()
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              framealpha=0.9, edgecolor="#ccc")

    ax.set_title("Yachay Tech — Topografía, Curvas de Nivel y Características del Campus",
                 **TITLE_FONT)
    ax.set_xlabel("Este (m, UTM Zona 17S)", **LABEL_FONT)
    ax.set_ylabel("Norte (m, UTM Zona 17S)", **LABEL_FONT)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, "02_contour_hillshade")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Shade Analysis
# ══════════════════════════════════════════════════════════════════════════════
def plot_shade():
    print("── Shade analysis …")
    dem, xs, ys = _load_dem_utm()

    hs = _hillshade(dem, azimuth=SUN_AZIMUTH, altitude=SUN_ALTITUDE)
    terrain_shade = (hs < 0.35).astype(float)

    blds = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)
    building_shadow = np.zeros_like(dem)

    if blds is not None and len(blds):
        shadow_len = 6.0 / np.tan(np.radians(SUN_ALTITUDE))
        shadow_dx  = -shadow_len * np.sin(np.radians(SUN_AZIMUTH))
        shadow_dy  = -shadow_len * np.cos(np.radians(SUN_AZIMUTH))
        from shapely.affinity import translate
        from rasterio.features import rasterize
        shadows = [translate(g, shadow_dx, shadow_dy)
                   for g in blds.geometry if g is not None and not g.is_empty]
        shadow_gdf = gpd.GeoDataFrame(geometry=shadows, crs=EPSG_LOCAL)
        transform = from_bounds(xs.min(), ys.min(), xs.max(), ys.max(),
                                 dem.shape[1], dem.shape[0])
        building_shadow = rasterize(
            [(g, 1) for g in shadow_gdf.geometry if g is not None and not g.is_empty],
            out_shape=dem.shape, transform=transform, fill=0, dtype="float32")

    shade_combined = np.clip(terrain_shade + building_shadow, 0, 1)

    # Save for scoring
    transform = from_bounds(xs.min(), ys.min(), xs.max(), ys.max(),
                             dem.shape[1], dem.shape[0])
    with rasterio.open(PROC / "shade_combined.tif", "w", driver="GTiff",
                        height=dem.shape[0], width=dem.shape[1], count=1,
                        dtype="float32", crs=CRS.from_epsg(EPSG_LOCAL),
                        transform=transform) as ds:
        ds.write(shade_combined.astype("float32"), 1)

    # ── 3-panel figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), facecolor="white")

    panels = [
        (terrain_shade, "Sombra del Terreno",     "Purples", "Sombra (0–1)"),
        (building_shadow, "Sombra de Edificaciones", "Blues",  "Sombra (0–1)"),
        (shade_combined, "Índice de Sombra Combinado", "RdPu", "Sombra (0–1)"),
    ]

    for ax, (data, title, cmap, clabel) in zip(axes, panels):
        im = ax.imshow(data, extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                       cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="equal")
        if blds is not None and len(blds):
            blds.plot(ax=ax, facecolor="#fce8c8", edgecolor="#8b5e3c",
                      linewidth=0.7, alpha=0.85, zorder=3)
        _zone_circles(ax, xs, ys)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
        cb.set_label(clabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Este (m)", fontsize=8)
        ax.set_ylabel("Norte (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        _add_scalebar(ax)
        _north_arrow(ax)
        ax.grid(True, alpha=0.2, linestyle="--")

    fig.suptitle(
        f"Yachay Tech — Análisis de Sombra  "
        f"(Azimut solar={SUN_AZIMUTH}°, Altitud solar={SUN_ALTITUDE}°)",
        fontsize=13, fontweight="bold", y=1.01)

    # Shared zone legend below
    legend_items = _zone_legend()
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    _save(fig, "03_shade_analysis")
    print(f"   Shade raster → {PROC / 'shade_combined.tif'}")


# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, name):
    for ext in ("png", "pdf"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   Saved: {OUT / name}.png")


if __name__ == "__main__":
    plot_terrain_3d()
    plot_contours()
    plot_shade()
    print("\n✓ Terrain visualizations complete.")
