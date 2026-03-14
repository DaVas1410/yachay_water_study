"""Generate clearer, decision-oriented visualizations."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pyproj import Transformer

OUT.mkdir(parents=True, exist_ok=True)


def _load_vec(path, layer=None, crs=None):
    try:
        gdf = gpd.read_file(path, layer=layer)
        return gdf.to_crs(crs) if crs else gdf
    except Exception:
        return None


def _decision_map():
    cand = pd.read_csv(PROC / "candidates.csv")
    blds = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)
    roads = _load_vec(RAW / "roads.gpkg", "roads", EPSG_LOCAL)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
    ax.set_facecolor("#f7fbff")

    if roads is not None and len(roads):
        roads.plot(ax=ax, color="#9e9e9e", linewidth=0.45, alpha=0.55, zorder=1)
    if blds is not None and len(blds):
        blds.plot(ax=ax, facecolor="#e8f0e8", edgecolor="#5f7f5f", linewidth=0.6, alpha=0.9, zorder=2)

    # plot required anchors
    tr = Transformer.from_crs(4326, EPSG_LOCAL, always_xy=True)
    all_anchors = REQUIRED_ANCHORS + ZONE_MANDATORY_ANCHORS
    for a in all_anchors:
        ex, ny = tr.transform(a["lon"], a["lat"])
        ax.plot(ex, ny, marker="x", color="#212121", markersize=8, mew=2, zorder=4)

    # plot candidates
    for _, r in cand.iterrows():
        color = "#2e7d32" if bool(r.get("in_building", False)) else "#c62828"
        marker = "s" if bool(r.get("in_building", False)) else "^"
        ax.plot(r.easting, r.northing, marker=marker, color=color, markersize=11,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        txt = f"#{int(r['Rango'])} {str(r.get('anchor_label',''))}"
        ax.annotate(txt, (r.easting, r.northing), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="#1a237e", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.set_title("Mapa de Decisión de Estaciones de Agua", fontsize=14, fontweight="bold")
    ax.set_xlabel("Este (m, UTM 17S)")
    ax.set_ylabel("Norte (m, UTM 17S)")
    ax.grid(True, alpha=0.2, linestyle="--")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='x', color='k', linestyle='None', markersize=8, label='Sitio requerido (entrada)'),
        Line2D([0],[0], marker='s', color='#2e7d32', linestyle='None', markersize=9, label='Candidato dentro de edificio'),
        Line2D([0],[0], marker='^', color='#c62828', linestyle='None', markersize=9, label='Candidato adyacente'),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT / "07_decision_map.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "07_decision_map.pdf", bbox_inches="tight")
    plt.close(fig)


def _solar_trend():
    df = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    labels = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    vals = [float(df.loc[m, "ALLSKY_SFC_SW_DWN"]) for m in months]
    annual = float(df.loc["Ann", "ALLSKY_SFC_SW_DWN"])

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.plot(labels, vals, marker="o", color="#ef6c00", linewidth=2.5, label="GHI mensual")
    ax.axhline(annual, color="#1a237e", linestyle="--", linewidth=1.8, label=f"Promedio anual ({annual:.2f})")
    max_i = int(np.argmax(vals)); min_i = int(np.argmin(vals))
    ax.scatter([labels[max_i]], [vals[max_i]], color="#c62828", s=70, zorder=5)
    ax.scatter([labels[min_i]], [vals[min_i]], color="#1565c0", s=70, zorder=5)
    ax.annotate("Mes más alto", (labels[max_i], vals[max_i]), xytext=(0,10), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate("Mes más bajo", (labels[min_i], vals[min_i]), xytext=(0,-14), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title("Tendencia de Radiación Solar Mensual (GHI)", fontsize=13, fontweight="bold")
    ax.set_ylabel("kWh/m²/día")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "08_solar_trend.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "08_solar_trend.pdf", bbox_inches="tight")
    plt.close(fig)


def _candidate_quality():
    p = BASE / "outputs" / "report_data" / "candidates_geospatial.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("Rango")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    colors = ["#2e7d32" if b else "#c62828" for b in df["in_building"]]
    axes[0].bar(df["Rango"].astype(str), df["score"], color=colors)
    axes[0].set_title("Puntaje por Candidato")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Puntaje (0-1)")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(df["Rango"].astype(str), df["dist_to_road_m"], color="#546e7a")
    axes[1].set_title("Distancia a Ruta Estudiantil")
    axes[1].set_ylabel("Metros a vía/sendero")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Resumen de Calidad de Candidatos", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "09_candidate_quality.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "09_candidate_quality.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # import anchors from current scoring script without circular imports at module load
    import importlib.util
    s = importlib.util.spec_from_file_location("scripts04", str(BASE / "scripts" / "04_scoring.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    globals()["REQUIRED_ANCHORS"] = m.REQUIRED_ANCHORS
    globals()["ZONE_MANDATORY_ANCHORS"] = m.ZONE_MANDATORY_ANCHORS

    _decision_map()
    _solar_trend()
    _candidate_quality()
    print("✓ Intuitive figures generated")
