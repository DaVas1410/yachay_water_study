"""
Script 04: Multi-factor Scoring, Building Prioritization & Cluster-based
           Candidate Selection for Water Refilling Stations.

Layers
------
1. Solar exposure   (W_SOLAR    = 30%) – high radiation → higher water demand
2. Shade index      (W_SHADE    = 20%) – shaded spots → comfortable placement
3. Building interior(W_BUILDING = 35%) – must be inside/near a building to install
4. Pedestrian prox. (W_PROXIMITY= 15%) – near paths/roads → foot traffic

Candidate constraints
---------------------
• Stakeholder-provided required sites are used as anchor points.
• Each anchor is snapped to the best local pixel, prioritizing inside-building
  placements when possible.
• Mandatory zone coverage is enforced for Jardín Botánico and Innopolis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point
from pyproj import Transformer

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.2, "grid.linestyle": "--",
})

# User-constrained priority sites (provided by campus stakeholders)
REQUIRED_ANCHORS = [
    {"label": "Biblioteca", "lat": 0.4051175903465887, "lon": -78.1741360644638, "count": 1},
    {"label": "Labs", "lat": 0.40389590543579784, "lon": -78.17244073251689, "count": 1},
    {"label": "Sport Fields", "lat": 0.4023889810638366, "lon": -78.17231156436854, "count": 1},
    {"label": "Multifamiliares", "lat": 0.4026957478287291, "lon": -78.17507791552728, "count": 1},
    {"label": "Caballerizas", "lat": 0.4045801458845306, "lon": -78.17591861919236, "count": 1},
    {"label": "SENESCYT", "lat": 0.4068123148712263, "lon": -78.17021432201662, "count": 2},
]

ZONE_MANDATORY_ANCHORS = [
    {"label": "Jardin Botanico", "lat": 0.4183386970866529, "lon": -78.18790548627916, "count": 1},
    {"label": "Innopolis", "lat": 0.41984081733076445, "lon": -78.18990836679563, "count": 1},
]

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
    for z in ZONES.values():
        cx, cy = _t4326_to_utm(z["lat"], z["lon"])
        r = z["radius_m"]
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor=z["color"],
                                 linewidth=1.8, linestyle="--", zorder=6))
        ax.annotate(z["name"], (cx, cy + r * 1.1), ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=z["color"], zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

def _load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        t    = src.transform
        r, c = data.shape
        xs = np.array([t.c + t.a * j for j in range(c)])
        ys = np.array([t.f + t.e * i for i in range(r)])
    return data, xs, ys

def _normalize(arr, invert=False):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr)
    n = (arr - mn) / (mx - mn)
    return 1 - n if invert else n

def _load_vec(path, layer=None, crs=None):
    try:
        gdf = gpd.read_file(path, layer=layer)
        return gdf.to_crs(crs) if crs else gdf
    except Exception:
        return None

def _save_raster(data, xs, ys, path):
    transform = from_bounds(xs.min(), ys.min(), xs.max(), ys.max(),
                             data.shape[1], data.shape[0])
    with rasterio.open(path, "w", driver="GTiff",
                        height=data.shape[0], width=data.shape[1],
                        count=1, dtype="float32",
                        crs=CRS.from_epsg(EPSG_LOCAL), transform=transform) as ds:
        ds.write(data.astype("float32"), 1)

def _dem_grid():
    """Return (rows, cols, xs, ys, transform, pixel_size_m)."""
    with rasterio.open(PROC / "dem_utm17s.tif") as src:
        rows, cols = src.height, src.width
        t = src.transform
        xs = np.array([t.c + t.a * j for j in range(cols)])
        ys = np.array([t.f + t.e * i for i in range(rows)])
        pxsz = abs(t.a)
    transform = from_bounds(xs.min(), ys.min(), xs.max(), ys.max(), cols, rows)
    return rows, cols, xs, ys, transform, pxsz

def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {OUT / name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Scoring layers
# ══════════════════════════════════════════════════════════════════════════════
def score_solar():
    """High solar radiation → high water demand → high score."""
    data, xs, ys = _load_raster(PROC / "solar_annual.tif")
    score = _normalize(data)
    _save_raster(score, xs, ys, PROC / "score_solar.tif")
    return score, xs, ys

def score_shade():
    """More shade → more comfortable placement → high score."""
    data, xs, ys = _load_raster(PROC / "shade_combined.tif")
    score = _normalize(data)
    _save_raster(score, xs, ys, PROC / "score_shade.tif")
    return score, xs, ys

def score_building_interior():
    """
    Pixels INSIDE building footprints get score = 1.0.
    Score decays exponentially with distance from nearest building
    (decay sigma = 50 m), so stations must be physically installable.
    Returns: score array, xs, ys, binary building_mask
    """
    rows, cols, xs, ys, transform, pxsz = _dem_grid()
    blds = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)

    building_mask = np.zeros((rows, cols), dtype="float32")
    if blds is not None and len(blds):
        building_mask = rasterize(
            [(g, 1) for g in blds.geometry if g is not None and not g.is_empty],
            out_shape=(rows, cols), transform=transform, fill=0, dtype="float32",
        )

    dist_px   = distance_transform_edt(1 - building_mask)
    dist_m    = dist_px * pxsz
    score     = np.exp(-dist_m / 50.0)   # sigma = 50 m
    _save_raster(score, xs, ys, PROC / "score_building.tif")
    return score, xs, ys, building_mask

def score_pedestrian():
    """Close to roads/footpaths → foot traffic → high score."""
    rows, cols, xs, ys, transform, pxsz = _dem_grid()
    occupied = np.zeros((rows, cols), dtype="float32")
    for layer, fname in [("roads", "roads.gpkg"), ("footpaths", "footpaths.gpkg")]:
        gdf = _load_vec(RAW / fname, layer, EPSG_LOCAL)
        if gdf is not None and len(gdf):
            occupied = np.clip(occupied + rasterize(
                [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty],
                out_shape=(rows, cols), transform=transform, fill=0, dtype="float32"), 0, 1)
    dist = distance_transform_edt(1 - occupied)
    score = _normalize(dist, invert=True)
    _save_raster(score, xs, ys, PROC / "score_proximity.tif")
    return score, xs, ys

def score_combined():
    print("── Computing scoring layers …")
    s_solar,         xs, ys = score_solar()
    s_shade,          _,  _ = score_shade()
    s_bld, _, _, bld_mask   = score_building_interior()
    s_prox,           _,  _ = score_pedestrian()

    composite = (W_SOLAR     * s_solar
               + W_SHADE     * s_shade
               + W_BUILDING  * s_bld
               + W_PROXIMITY * s_prox)
    _save_raster(composite, xs, ys, PROC / "score_combined.tif")
    print(f"   Score range: {composite.min():.3f} – {composite.max():.3f}")
    return composite, s_solar, s_shade, s_bld, s_prox, bld_mask, xs, ys


# ══════════════════════════════════════════════════════════════════════════════
# Constrained candidate extraction (stakeholder-defined anchors)
# ══════════════════════════════════════════════════════════════════════════════
def _select_candidates_from_anchor(score, bld_mask, xs, ys, label, lat, lon, count=1,
                                   search_radius_m=260, min_separation_m=85,
                                   forbidden_cells=None):
    """
    Snap candidate(s) to the best scoring pixels near a required anchor.
    Preference order:
    1) inside building within radius
    2) highest score within radius
    """
    ex, ny = _t4326_to_utm(lat, lon)
    XX, YY = np.meshgrid(xs, ys)
    dist = np.sqrt((XX - ex) ** 2 + (YY - ny) ** 2)
    local_mask = (dist <= search_radius_m) & np.isfinite(score)
    ri, ci = np.where(local_mask)
    if len(ri) == 0:
        return []

    local_scores = score[ri, ci]
    local_bld    = bld_mask[ri, ci]
    local_dist   = dist[ri, ci]
    score_norm   = (local_scores - np.nanmin(local_scores)) / (np.nanmax(local_scores) - np.nanmin(local_scores) + 1e-9)
    dist_norm    = local_dist / max(search_radius_m, 1)
    # Bias to stay near required anchor while keeping high score
    objective    = 0.72 * score_norm + 0.28 * (1 - dist_norm)
    order = np.argsort(-objective)

    chosen = []
    used_xy = []
    for idx in order:
        r, c = int(ri[idx]), int(ci[idx])
        x, y = float(xs[c]), float(ys[r])
        inside = bool(local_bld[idx] > 0.5)
        if forbidden_cells is not None and (r, c) in forbidden_cells:
            continue

        # Enforce minimum spacing for multi-point anchors (e.g., SENESCYT x2)
        too_close = any(np.hypot(x - ux, y - uy) < min_separation_m for ux, uy in used_xy)
        if too_close:
            continue

        chosen.append({
            "anchor_label": label,
            "row": r, "col": c,
            "easting": x, "northing": y,
            "score": float(score[r, c]),
            "in_building": inside,
            "cluster_size": int(len(ri)),
            "anchor_lat": lat,
            "anchor_lon": lon,
        })
        used_xy.append((x, y))
        if forbidden_cells is not None:
            forbidden_cells.add((r, c))
        if len(chosen) >= count:
            break

    # If no inside-building point selected, fallback best overall in radius
    if len(chosen) < count:
        for idx in order:
            r, c = int(ri[idx]), int(ci[idx])
            x, y = float(xs[c]), float(ys[r])
            if forbidden_cells is not None and (r, c) in forbidden_cells:
                continue
            if any(np.isclose(x, ch["easting"]) and np.isclose(y, ch["northing"]) for ch in chosen):
                continue
            chosen.append({
                "anchor_label": label,
                "row": r, "col": c,
                "easting": x, "northing": y,
                "score": float(score[r, c]),
                "in_building": bool(local_bld[idx] > 0.5),
                "cluster_size": int(len(ri)),
                "anchor_lat": lat,
                "anchor_lon": lon,
            })
            if forbidden_cells is not None:
                forbidden_cells.add((r, c))
            if len(chosen) >= count:
                break

    return chosen


def _build_constrained_candidates(score, bld_mask, xs, ys):
    """
    Build candidates from user-required sites plus mandatory zone coverage.
    Excludes unconstrained outliers (e.g., Urcuqui town) by design.
    """
    all_anchors = REQUIRED_ANCHORS + ZONE_MANDATORY_ANCHORS
    records = []
    forbidden_cells = set()
    for a in all_anchors:
        records.extend(
            _select_candidates_from_anchor(
                score, bld_mask, xs, ys,
                label=a["label"], lat=a["lat"], lon=a["lon"], count=a["count"],
                search_radius_m=280 if a["label"] == "SENESCYT" else 240,
                min_separation_m=95 if a["label"] == "SENESCYT" else 80,
                forbidden_cells=forbidden_cells,
            )
        )

    # Rank globally by score, but keep all required groups represented
    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["Rango"] = np.arange(1, len(df) + 1)
    return df


def _enrich_with_building_info(df):
    """Spatial join: find building name/type for each candidate."""
    blds = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)
    if blds is None:
        df["building_name"] = "—"
        df["building_type"] = "—"
        return df

    # Keep useful cols
    keep = ["geometry", "name", "building", "amenity"]
    blds_s = blds[[c for c in keep if c in blds.columns]].copy()
    blds_s["_bid"] = range(len(blds_s))

    pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(r.easting, r.northing) for _, r in df.iterrows()],
        crs=EPSG_LOCAL,
    )
    joined = gpd.sjoin(pts, blds_s, how="left", predicate="within")

    # Deduplicate (take first match per candidate)
    joined = joined.groupby(joined.index).first()

    df["building_name"] = joined.get("name", "—").fillna("—").values
    df["building_type"] = joined.get("building", "—").fillna(
                          joined.get("amenity", "—")).fillna("—").values
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════
def plot_candidates():
    print("── Building scoring maps + constrained candidates …")
    composite, s_solar, s_shade, s_bld, s_prox, bld_mask, xs, ys = score_combined()

    candidates = _build_constrained_candidates(composite, bld_mask, xs, ys)
    candidates = _enrich_with_building_info(candidates)

    # Add lat/lon
    tr = Transformer.from_crs(EPSG_LOCAL, 4326, always_xy=True)
    lons, lats = tr.transform(candidates.easting.values, candidates.northing.values)
    candidates["Latitud"]  = [f"{v:.6f}" for v in lats]
    candidates["Longitud"] = [f"{v:.6f}" for v in lons]
    candidates.to_csv(PROC / "candidates.csv", index=False)

    print(candidates[["Rango","Latitud","Longitud","score",
                       "in_building","building_name","building_type"]].to_string(index=False))

    blds  = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)
    roads = _load_vec(RAW / "roads.gpkg", "roads", EPSG_LOCAL)

    cmap_score = LinearSegmentedColormap.from_list(
        "score", ["#d32f2f", "#ff7043", "#ffca28", "#66bb6a", "#1b5e20"])

    layer_specs = [
        (s_solar,  f"Radiación Solar  ({int(W_SOLAR*100)}%)",      "YlOrRd"),
        (s_shade,  f"Sombra  ({int(W_SHADE*100)}%)",               "Blues"),
        (s_bld,    f"Interior de Edificaciones  ({int(W_BUILDING*100)}%)", "Oranges"),
        (composite, "Puntuacion Combinada + Candidatos Restringidos",       cmap_score),
    ]

    # ── 4-panel scoring grid ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 19), facecolor="white")
    gs  = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.42],
                            hspace=0.42, wspace=0.28,
                            left=0.07, right=0.97, top=0.94, bottom=0.03)
    map_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    rank_colors = ["#c62828","#e65100","#ef6c00","#1b5e20",
                   "#006064","#4a148c","#880e4f","#37474f"]

    for ax, (data, title, cmap) in zip(map_axes, layer_specs):
        ax.set_facecolor("#e8f0f7")
        im = ax.imshow(data, extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                       cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="equal")

        if blds is not None and len(blds):
            blds.plot(ax=ax, facecolor="none", edgecolor="#555",
                      linewidth=0.65, alpha=0.7, zorder=3)
        if roads is not None and len(roads):
            roads.plot(ax=ax, color="#777", linewidth=0.35, alpha=0.4, zorder=2)

        _zone_circles(ax)

        # Show constrained candidates on combined panel
        if "Restringidos" in title:
            # Draw local influence halos
            for _, row in candidates.iterrows():
                rk = int(row["Rango"])
                rc = rank_colors[(rk-1) % len(rank_colors)]
                # Local halo near each constrained anchor
                halo = plt.Circle((row.easting, row.northing),
                                   120,
                                   color=rc, alpha=0.12, zorder=4)
                ax.add_patch(halo)
                halo_edge = plt.Circle((row.easting, row.northing),
                                        120,
                                        fill=False, edgecolor=rc,
                                        linewidth=1.2, linestyle=":", zorder=5)
                ax.add_patch(halo_edge)

            # Highlight buildings that contain candidates
            if blds is not None:
                from shapely.geometry import Point as ShPoint
                in_bld_rows = candidates[candidates["in_building"] == True]
                bld_highlight_idx = []
                for _, row in in_bld_rows.iterrows():
                    pt = ShPoint(row.easting, row.northing)
                    for i2, br in blds.iterrows():
                        if br.geometry is not None and br.geometry.contains(pt):
                            bld_highlight_idx.append(i2)
                            break
                if bld_highlight_idx:
                    blds.loc[bld_highlight_idx].plot(
                        ax=ax, facecolor="#fff176", edgecolor="#f57f17",
                        linewidth=1.5, alpha=0.9, zorder=4)

            # Candidate markers
            for _, row in candidates.iterrows():
                rk = int(row["Rango"])
                rc = rank_colors[(rk-1) % len(rank_colors)]
                marker = "s" if row["in_building"] else "^"
                ax.plot(row.easting, row.northing, marker, markersize=13,
                        color=rc, zorder=9, markeredgecolor="white",
                        markeredgewidth=1.8)
                ax.annotate(f"#{rk}", (row.easting, row.northing),
                            ha="center", va="center", fontsize=6.5,
                            fontweight="bold", color="white", zorder=10)

        cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, shrink=0.80)
        cb.set_label("Puntuacion (0–1)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
        ax.set_xlabel("Este (m)", fontsize=8)
        ax.set_ylabel("Norte (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        _add_scalebar(ax)
        _north_arrow(ax)

    # ── Candidate table (bottom row) ─────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")

    tbl_data = [[
        f"#{int(r['Rango'])}",
        str(r["anchor_label"]),
        lats[i].__format__(".6f") + "°",
        lons[i].__format__(".6f") + "°",
        f"{r.easting:.1f}",
        f"{r.northing:.1f}",
        f"{r['score']:.4f}",
        "Si" if r["in_building"] else "No",
        str(r["building_name"])[:28],
        str(r["building_type"]),
        str(r["cluster_size"]),
    ] for i, (_, r) in enumerate(candidates.iterrows())]

    col_labels = ["Rango","Sitio","Latitud","Longitud","Este (m)","Norte (m)",
                  "Punt.","En Edificio","Nombre Edificio","Tipo","Px/Busqueda"]
    tbl = ax_tbl.table(cellText=tbl_data, colLabels=col_labels,
                        cellLoc="center", loc="center",
                        bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r2, c2), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r2 == 0:
            cell.set_facecolor("#1a3a5c")
            cell.set_text_props(color="white", fontweight="bold")
        elif r2 <= len(tbl_data):
            rc = rank_colors[(r2-1) % len(rank_colors)]
            # Rango column
            if c2 == 0:
                cell.set_facecolor(rc)
                cell.set_text_props(color="white", fontweight="bold")
            # "En Edificio" column — green/red
            elif c2 == 7:
                val = tbl_data[r2-1][7]
                cell.set_facecolor("#c8e6c9" if val == "Si" else "#ffcdd2")
                cell.set_text_props(fontweight="bold")
            elif r2 % 2 == 0:
                cell.set_facecolor("#f5f5f5")

    ax_tbl.set_title("Tabla 1 — Candidatos Restringidos por Sitio Prioritario  |  "
                     "Cuadrado = dentro de edificio  |  Triangulo = fuera (adyacente)",
                     fontsize=11, fontweight="bold", pad=8)

    # Legend for markers
    legend_els = [
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#1b5e20",
               markersize=10, markeredgecolor="white", label="Dentro de edificio"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#c62828",
               markersize=10, markeredgecolor="white", label="Fuera de edificio"),
        mpatches.Patch(facecolor="#fff176", edgecolor="#f57f17",
                       label="Edificio candidato (resaltado)"),
    ] + [mpatches.Patch(facecolor=z["light"], edgecolor=z["color"],
                         linewidth=1.5, label=z["name"])
         for z in ZONES.values()]
    fig.legend(handles=legend_els, loc="lower center", ncol=6,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.01),
               framealpha=0.95, edgecolor="#ccc")

    fig.suptitle(
        "Yachay Tech — Analisis Multi-criterio con Priorizacion de Edificaciones\n"
        f"Sitios requeridos + cobertura de zonas (Jardin Botanico e Innopolis)  |  "
        f"Solar {int(W_SOLAR*100)}% · Sombra {int(W_SHADE*100)}% · "
        f"Edificacion {int(W_BUILDING*100)}% · Proximidad {int(W_PROXIMITY*100)}%",
        fontsize=13, fontweight="bold")

    _save(fig, "06_scoring_candidates")
    return candidates


if __name__ == "__main__":
    plot_candidates()
    print("\n✓ Scoring + clustering complete.")
