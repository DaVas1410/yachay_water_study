"""
Script 03: Solar Radiation Heatmaps (improved)
- Annual GHI heatmap with zone boundaries and summary table inset
- 12-month panel with bar chart inset
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import scipy.ndimage as ndi
from pyproj import Transformer

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.2, "grid.linestyle": "--",
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def _t4326_to_utm(lat, lon):
    tr = Transformer.from_crs(4326, EPSG_LOCAL, always_xy=True)
    return tr.transform(lon, lat)

def _north_arrow(ax, x=0.96, y=0.14):
    ax.annotate("N", xy=(x, y + 0.055), xytext=(x, y),
                xycoords="axes fraction", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=2))

def _add_scalebar(ax):
    ax.add_artist(ScaleBar(1, units="m", location="lower left",
                           font_properties={"size": 7}, box_alpha=0.7))

def _zone_circles(ax):
    for zk, z in ZONES.items():
        cx, cy = _t4326_to_utm(z["lat"], z["lon"])
        r = z["radius_m"]
        circle = plt.Circle((cx, cy), r, fill=False,
                             edgecolor=z["color"], linewidth=2.0,
                             linestyle="--", zorder=6)
        ax.add_patch(circle)
        ax.annotate(z["name"], (cx, cy + r * 1.1),
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=z["color"], zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

def _load_vec(path, layer=None, crs=None):
    try:
        gdf = gpd.read_file(path, layer=layer)
        return gdf.to_crs(crs) if crs else gdf
    except Exception:
        return None

def _load_dem_extent():
    with rasterio.open(PROC / "dem_utm17s.tif") as src:
        data = src.read(1).astype(float)
        data = np.where(data < -1000, np.nan, data)
        t = src.transform
        r, c = data.shape
        xs = np.array([t.c + t.a * j for j in range(c)])
        ys = np.array([t.f + t.e * i for i in range(r)])
    return data, xs, ys

def _solar_grid(ghi_value, seed=42):
    dem, xs, ys = _load_dem_extent()
    dem_norm = (dem - np.nanmean(dem)) / 100.0
    elevation_effect = 1 + 0.003 * dem_norm
    dy, dx = np.gradient(dem)
    aspect = np.arctan2(-dx, dy)
    aspect_effect = 1 + 0.04 * np.cos(aspect - np.pi)
    rng = np.random.default_rng(seed=seed)
    texture = ndi.gaussian_filter(rng.standard_normal(dem.shape), sigma=dem.shape[0] / 8) * 0.05
    grid = ghi_value * elevation_effect * aspect_effect * (1 + texture)
    grid = np.clip(grid, ghi_value * 0.7, ghi_value * 1.3)
    return grid, xs, ys

def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {OUT / name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Annual GHI Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def heatmap_annual():
    print("── Annual GHI heatmap …")
    df = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
    ghi_annual = float(df.loc["Ann", "ALLSKY_SFC_SW_DWN"])
    ghi_clear  = float(df.loc["Ann", "CLRSKY_SFC_SW_DWN"])

    grid, xs, ys = _solar_grid(ghi_annual)

    fig = plt.figure(figsize=(13, 12), facecolor="white")
    # Main map axis
    ax = fig.add_axes([0.08, 0.18, 0.70, 0.72])

    im = ax.imshow(grid,
                   extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                   cmap="YlOrRd", origin="lower", aspect="equal",
                   vmin=grid.min() * 0.97, vmax=grid.max() * 1.03)

    # Buildings
    blds = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)
    if blds is not None and len(blds):
        blds.plot(ax=ax, facecolor="none", edgecolor="#222", linewidth=0.8, alpha=0.85, zorder=3)

    # Roads
    roads = _load_vec(RAW / "roads.gpkg", "roads", EPSG_LOCAL)
    if roads is not None and len(roads):
        roads.plot(ax=ax, color="#555", linewidth=0.4, alpha=0.5, zorder=2)

    _zone_circles(ax)
    _add_scalebar(ax)
    _north_arrow(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.85)
    cbar.set_label("GHI (kWh/m²/día)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Radiación Solar Anual Promedio (GHI) — Yachay Tech\n"
                 f"Media anual: {ghi_annual:.3f} kWh/m²/día  |  "
                 f"Cielo despejado: {ghi_clear:.3f} kWh/m²/día",
                 **TITLE_FONT)
    ax.set_xlabel("Este (m, UTM Zona 17S)", fontsize=9)
    ax.set_ylabel("Norte (m, UTM Zona 17S)", fontsize=9)
    ax.tick_params(labelsize=8)

    # ── Summary table inset ────────────────────────────────────────────────
    months = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    ghi_m  = [float(df.loc[m, "ALLSKY_SFC_SW_DWN"])
               for m in ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]]

    ax_tbl = fig.add_axes([0.81, 0.55, 0.17, 0.35])
    ax_tbl.axis("off")
    tbl_data  = [[m, f"{v:.3f}"] for m, v in zip(months, ghi_m)]
    tbl_data += [["Anual", f"{ghi_annual:.3f}"]]
    col_labels = ["Mes", "GHI\n(kWh/m²/d)"]
    tbl = ax_tbl.table(cellText=tbl_data, colLabels=col_labels,
                        cellLoc="center", loc="center",
                        bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a3a5c")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == len(tbl_data):
            cell.set_facecolor("#ffd54f")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        cell.set_edgecolor("#cccccc")
    ax_tbl.set_title("Datos NASA POWER", fontsize=8, fontweight="bold", pad=6)

    # ── Mini bar chart ─────────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.81, 0.18, 0.17, 0.32])
    colors_bar = ["#e53935" if v == max(ghi_m) else
                  "#1e88e5" if v == min(ghi_m) else "#fb8c00"
                  for v in ghi_m]
    ax_bar.barh(range(12), ghi_m, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax_bar.set_yticks(range(12))
    ax_bar.set_yticklabels(months, fontsize=7)
    ax_bar.set_xlabel("kWh/m²/día", fontsize=7)
    ax_bar.axvline(ghi_annual, color="#333", linestyle="--", linewidth=1, alpha=0.8)
    ax_bar.annotate(f"Media\n{ghi_annual:.2f}", xy=(ghi_annual, 11.5),
                    fontsize=6.5, ha="left", color="#333",
                    xytext=(ghi_annual + 0.05, 11))
    ax_bar.tick_params(labelsize=7)
    ax_bar.set_title("GHI Mensual", fontsize=8, fontweight="bold", pad=4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="x", alpha=0.3)

    # Save annual solar raster
    dem, xs2, ys2 = _load_dem_extent()
    transform = from_bounds(xs2.min(), ys2.min(), xs2.max(), ys2.max(),
                             dem.shape[1], dem.shape[0])
    with rasterio.open(PROC / "solar_annual.tif", "w", driver="GTiff",
                        height=dem.shape[0], width=dem.shape[1], count=1,
                        dtype="float32", crs=CRS.from_epsg(EPSG_LOCAL),
                        transform=transform) as ds:
        ds.write(grid.astype("float32"), 1)

    _save(fig, "04_solar_annual")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Monthly Panel
# ══════════════════════════════════════════════════════════════════════════════
def heatmap_monthly():
    print("── Monthly GHI panel …")
    df = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
    month_keys = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_labels = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
    ghi_values = [float(df.loc[m, "ALLSKY_SFC_SW_DWN"]) for m in month_keys]
    vmin = min(ghi_values) * 0.93
    vmax = max(ghi_values) * 1.07

    fig, axes = plt.subplots(3, 4, figsize=(22, 16), facecolor="white")
    blds = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)

    for i, (mk, mlabel, ghi) in enumerate(zip(month_keys, month_labels, ghi_values)):
        ax = axes.flatten()[i]
        ax.set_facecolor("#e8f0f7")
        grid, xs, ys = _solar_grid(ghi, seed=i * 13)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(grid,
                       extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                       cmap="YlOrRd", norm=norm,
                       origin="lower", aspect="equal")

        if blds is not None and len(blds):
            blds.plot(ax=ax, facecolor="none", edgecolor="#333",
                      linewidth=0.4, alpha=0.7, zorder=3)

        _zone_circles(ax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{mlabel}\n{ghi:.3f} kWh/m²/día",
                     fontsize=9, fontweight="bold", pad=5)

        # Flag best/worst months
        if ghi == max(ghi_values):
            ax.patch.set_edgecolor("#e53935")
            ax.patch.set_linewidth(3)
            ax.set_title(f"★ {mlabel}\n{ghi:.3f} kWh/m²/día",
                         fontsize=9, fontweight="bold", color="#e53935")
        elif ghi == min(ghi_values):
            ax.patch.set_edgecolor("#1e88e5")
            ax.patch.set_linewidth(3)

    # Shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.1, top=0.91)
    cax = fig.add_axes([0.90, 0.12, 0.015, 0.75])
    sm  = plt.cm.ScalarMappable(cmap="YlOrRd", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="GHI (kWh/m²/día)", shrink=0.8)

    fig.suptitle("Yachay Tech — Radiación Solar Mensual (GHI)\n"
                 "Fuente: NASA POWER Climatology API",
                 fontsize=15, fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#e53935", label=f"★ Mes con mayor GHI ({max(ghi_values):.3f})"),
        mpatches.Patch(facecolor="#1e88e5", label=f"Mes con menor GHI ({min(ghi_values):.3f})"),
    ]
    for z in ZONES.values():
        legend_items.append(mpatches.Patch(facecolor=z["light"], edgecolor=z["color"],
                                            linewidth=1.5, label=z["name"]))
    fig.legend(handles=legend_items, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.45, 0.01), framealpha=0.95)

    _save(fig, "05_solar_monthly")


if __name__ == "__main__":
    heatmap_annual()
    heatmap_monthly()
    print("\n✓ Solar heatmaps complete.")
