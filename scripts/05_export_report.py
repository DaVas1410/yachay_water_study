"""
Script 05: Interactive Map (improved) + PDF Report
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd
import rasterio
import folium
from folium import plugins, MacroElement
from jinja2 import Template
from pathlib import Path
from pyproj import Transformer

OUT_I.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_raster(path):
    with rasterio.open(path) as src:
        data  = src.read(1).astype(float)
        t     = src.transform
        r, c  = data.shape
        xs    = np.array([t.c + t.a * j for j in range(c)])
        ys    = np.array([t.f + t.e * i for i in range(r)])
    return data, xs, ys

def _load_vec(path, layer=None, crs=None):
    try:
        gdf = gpd.read_file(path, layer=layer)
        return gdf.to_crs(crs) if crs else gdf
    except Exception:
        return None

def _utm_to_wgs84(easting, northing):
    tr = Transformer.from_crs(EPSG_LOCAL, 4326, always_xy=True)
    lon, lat = tr.transform(easting, northing)
    return lat, lon

def _raster_to_heatmap(data, xs, ys, n=3000):
    """Sample pixels weighted by value for folium HeatMap."""
    flat = data.flatten()
    valid = np.isfinite(flat) & (flat > 0)
    fv = flat[valid]
    weights = fv / fv.sum()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(fv), size=min(n, len(fv)), replace=False, p=weights)
    ri, ci = np.unravel_index(np.where(valid)[0][idx], data.shape)
    tr = Transformer.from_crs(EPSG_LOCAL, 4326, always_xy=True)
    pts = []
    for r, c in zip(ri, ci):
        lon, lat = tr.transform(xs[c], ys[r])
        pts.append([lat, lon, float(data[r, c])])
    return pts


def export_geospatial_data_science_outputs():
    """
    Export machine-readable geospatial data-science outputs for reproducibility.
    """
    print("── Exporting geospatial data-science outputs …")
    out_dir = BASE / "outputs" / "report_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = PROC / "candidates.csv"
    if not cand_path.exists():
        print("   candidates.csv not found; skipping report_data exports.")
        return

    df = pd.read_csv(cand_path)
    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df.easting, df.northing),
        crs=EPSG_LOCAL,
    )

    # Distances to infrastructure
    roads = _load_vec(RAW / "roads.gpkg", "roads", EPSG_LOCAL)
    blds  = _load_vec(RAW / "buildings.gpkg", "buildings", EPSG_LOCAL)
    roads_union = roads.geometry.union_all() if roads is not None and len(roads) else None
    blds_union  = blds.geometry.union_all() if blds is not None and len(blds) else None
    pts["dist_to_road_m"] = pts.geometry.apply(
        lambda g: float(g.distance(roads_union)) if roads_union is not None else np.nan
    )
    pts["dist_to_building_m"] = pts.geometry.apply(
        lambda g: float(g.distance(blds_union)) if blds_union is not None else np.nan
    )

    # Zone assignment
    def nearest_zone(lat, lon):
        return min(ZONES.values(), key=lambda z: (z["lat"]-lat)**2 + (z["lon"]-lon)**2)["name"]
    tr = Transformer.from_crs(EPSG_LOCAL, 4326, always_xy=True)
    lons, lats = tr.transform(pts.easting.values, pts.northing.values)
    pts["lat"] = lats
    pts["lon"] = lons
    pts["nearest_zone"] = [nearest_zone(la, lo) for la, lo in zip(lats, lons)]

    # Save CSV + GeoJSON
    pts.drop(columns="geometry").to_csv(out_dir / "candidates_geospatial.csv", index=False)
    pts.to_crs(4326).to_file(out_dir / "candidates_geospatial.geojson", driver="GeoJSON")

    # Zone summary table
    zone_summary = (
        pts.groupby("nearest_zone")
          .agg(
              n_candidates=("Rango", "count"),
              n_inside_building=("in_building", "sum"),
              mean_score=("score", "mean"),
              mean_dist_to_road_m=("dist_to_road_m", "mean"),
          )
          .reset_index()
          .sort_values("n_candidates", ascending=False)
    )
    zone_summary.to_csv(out_dir / "zone_coverage_summary.csv", index=False)

    # Methodology metadata
    meta = {
        "study_area": "Yachay Tech + Jardin Botanico + Innopolis",
        "weights": {
            "solar": W_SOLAR,
            "shade": W_SHADE,
            "building": W_BUILDING,
            "proximity": W_PROXIMITY,
        },
        "candidate_constraints": [
            "Stakeholder-defined anchors",
            "Mandatory zones: Jardin Botanico, Innopolis",
            "Building-priority snapping",
        ],
        "outputs": [
            "candidates_geospatial.csv",
            "candidates_geospatial.geojson",
            "zone_coverage_summary.csv",
        ],
    }
    import json
    with open(out_dir / "methodology_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"   Saved report_data outputs in: {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive Folium Map — full redesign
# ══════════════════════════════════════════════════════════════════════════════
def export_interactive():
    print("── Building interactive folium map …")

    # Map centred on study area
    m = folium.Map(
        location=[STUDY_CENTER_LAT, STUDY_CENTER_LON],
        zoom_start=16,
        tiles=None,
        width="100%",
        height="100%",
    )

    # ── Basemaps ─────────────────────────────────────────────────────────────
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Satelital",
        show=True,
    ).add_to(m)
    folium.TileLayer(
        "OpenStreetMap",
        name="🗺️ OpenStreetMap",
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="📄 CartoDB Claro",
        show=False,
    ).add_to(m)

    # ── Heatmap layers ────────────────────────────────────────────────────────
    solar, xs, ys = _load_raster(PROC / "score_solar.tif")
    shade, _,  _  = _load_raster(PROC / "score_shade.tif")
    comp,  _,  _  = _load_raster(PROC / "score_combined.tif")

    def _heatmap_layer(name, data, gradient, show=True):
        fg = folium.FeatureGroup(name=name, show=show)
        plugins.HeatMap(
            _raster_to_heatmap(data, xs, ys),
            min_opacity=0.35, radius=20, blur=16,
            gradient=gradient,
        ).add_to(fg)
        return fg

    _heatmap_layer("☀️ Radiación Solar", solar,
                   {"0.3":"#fffde7","0.55":"#ff9800","1.0":"#b71c1c"},
                   show=True).add_to(m)
    _heatmap_layer("🌑 Índice de Sombra", shade,
                   {"0.3":"#e3f2fd","0.6":"#1565c0","1.0":"#0d0d40"},
                   show=False).add_to(m)
    _heatmap_layer("🏆 Puntuación Combinada", comp,
                   {"0.0":"#d32f2f","0.35":"#ff7043","0.6":"#ffca28",
                    "0.8":"#66bb6a","1.0":"#1b5e20"},
                   show=False).add_to(m)

    # ── Buildings ─────────────────────────────────────────────────────────────
    blds = _load_vec(RAW / "buildings.gpkg", "buildings")
    if blds is not None and len(blds):
        fg_bld = folium.FeatureGroup(name="🏢 Edificaciones", show=True)
        folium.GeoJson(
            blds.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#f4a460", "color": "#8b4513",
                "weight": 1.2, "fillOpacity": 0.45,
            },
            tooltip=folium.GeoJsonTooltip(fields=[], aliases=[]),
        ).add_to(fg_bld)
        fg_bld.add_to(m)

    # ── Roads ─────────────────────────────────────────────────────────────────
    roads = _load_vec(RAW / "roads.gpkg", "roads")
    if roads is not None and len(roads):
        fg_roads = folium.FeatureGroup(name="🛣️ Vías y Senderos", show=False)
        folium.GeoJson(
            roads.__geo_interface__,
            style_function=lambda _: {"color": "#607d8b", "weight": 1.5, "opacity": 0.65},
        ).add_to(fg_roads)
        fg_roads.add_to(m)

    # ── Zone circles ─────────────────────────────────────────────────────────
    fg_zones = folium.FeatureGroup(name="📍 Zonas del Campus", show=True)
    for zk, z in ZONES.items():
        folium.Circle(
            location=[z["lat"], z["lon"]],
            radius=z["radius_m"],
            color=z["color"], fill=True,
            fill_color=z["light"], fill_opacity=0.15,
            weight=2.5, dash_array="8 4",
            tooltip=z["name"],
            popup=folium.Popup(f"""
            <div style='font-family:sans-serif;padding:6px;min-width:160px'>
              <b style='color:{z["color"]};font-size:14px'>{z["name"]}</b><br>
              <hr style='margin:4px 0;border-color:#ddd'>
              <b>Lat:</b> {z["lat"]:.6f}°<br>
              <b>Lon:</b> {z["lon"]:.6f}°
            </div>""", max_width=220),
        ).add_to(fg_zones)
        # Zone label marker
        folium.Marker(
            location=[z["lat"], z["lon"]],
            icon=folium.DivIcon(
                html=f"""<div style='
                    background:{z["color"]};color:white;
                    font-weight:bold;font-size:11px;font-family:sans-serif;
                    padding:4px 8px;border-radius:12px;
                    box-shadow:1px 1px 4px rgba(0,0,0,0.4);
                    white-space:nowrap;
                '>{z["name"]}</div>""",
                icon_anchor=(0, 0),
            ),
        ).add_to(fg_zones)
    fg_zones.add_to(m)

    # ── Candidate water station markers ───────────────────────────────────────
    cand_path = PROC / "candidates.csv"
    if cand_path.exists():
        df = pd.read_csv(cand_path)
        solar_df = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
        ghi_ann = float(solar_df.loc["Ann", "ALLSKY_SFC_SW_DWN"])

        fg_cand = folium.FeatureGroup(name="💧 Estaciones de Agua Candidatas", show=True)

        rank_colors = ["#c62828","#e65100","#ef6c00","#1b5e20",
                       "#006064","#4a148c","#880e4f","#37474f"]

        for _, row in df.iterrows():
            lat_c, lon_c = _utm_to_wgs84(row.easting, row.northing)
            rk = int(row["Rango"])
            rc = rank_colors[(rk-1) % len(rank_colors)]

            nearest_zone = min(ZONES.values(),
                key=lambda z: (z["lat"]-lat_c)**2 + (z["lon"]-lon_c)**2)

            in_bld   = bool(row.get("in_building", False))
            bld_name = str(row.get("building_name", "—"))
            bld_type = str(row.get("building_type", "—"))
            anchor_label = str(row.get("anchor_label", "Sitio requerido"))
            bld_badge_color = "#2e7d32" if in_bld else "#c62828"
            bld_badge_text  = "DENTRO DEL EDIFICIO" if in_bld else "ADYACENTE AL EDIFICIO"
            # Use square marker for in-building, triangle-ish for outside
            marker_shape = "4px" if in_bld else "50% 50% 50% 0"

            popup_html = f"""
            <div style='font-family:"Segoe UI",sans-serif;min-width:240px;padding:4px'>
              <div style='background:{rc};color:white;padding:8px 12px;
                          border-radius:6px 6px 0 0;font-size:15px;font-weight:bold'>
                Candidato #{rk}
              </div>
              <div style='background:{bld_badge_color};color:white;padding:4px 12px;
                          font-size:10px;font-weight:bold;letter-spacing:0.5px'>
                {bld_badge_text}
              </div>
              <div style='padding:8px 12px;border:1px solid #ddd;
                          border-radius:0 0 6px 6px;background:#fafafa'>
                <table style='width:100%;font-size:12px;border-collapse:collapse'>
                  <tr><td style='color:#666;padding:2px 0'>Puntaje</td>
                      <td style='font-weight:bold;color:{rc}'>{row["score"]:.4f} / 1.0</td></tr>
                  <tr><td style='color:#666;padding:2px 0'>Sitio requerido</td>
                      <td style='font-weight:bold'>{anchor_label}</td></tr>
                  <tr><td style='color:#666;padding:2px 0'>Latitud</td>
                      <td>{lat_c:.6f} deg</td></tr>
                  <tr><td style='color:#666;padding:2px 0'>Longitud</td>
                      <td>{lon_c:.6f} deg</td></tr>
                  <tr><td style='color:#666;padding:2px 0'>Edificio</td>
                      <td style='font-weight:bold'>{bld_name}</td></tr>
                  <tr><td style='color:#666;padding:2px 0'>Tipo edificio</td>
                      <td>{bld_type}</td></tr>
                  <tr><td style='color:#666;padding:2px 0'>Zona</td>
                      <td style='color:{nearest_zone["color"]};font-weight:bold'>
                          {nearest_zone["name"]}</td></tr>
                  <tr><td style='color:#666;padding:2px 0'>GHI anual</td>
                      <td>{ghi_ann:.3f} kWh/m²/dia</td></tr>
                  <tr><td style='color:#666;padding:2px 0'>Cluster (pixeles)</td>
                      <td>{int(row.get("cluster_size", 0))}</td></tr>
                </table>
                <div style='margin-top:6px;font-size:9.5px;color:#999;
                             border-top:1px solid #eee;padding-top:5px'>
                  Solar {int(W_SOLAR*100)}% · Sombra {int(W_SHADE*100)}% · Edificacion {int(W_BUILDING*100)}% · Proximidad {int(W_PROXIMITY*100)}%
                </div>
              </div>
            </div>
            """
            folium.Marker(
                location=[lat_c, lon_c],
                popup=folium.Popup(popup_html, max_width=290),
                tooltip=f"Candidato #{rk} | Puntaje: {row['score']:.3f} | {'En edificio' if in_bld else 'Adyacente'}",
                icon=folium.DivIcon(
                    html=f"""<div style='
                        width:30px;height:30px;
                        background:{rc};color:white;
                        font-weight:bold;font-size:13px;font-family:sans-serif;
                        border-radius:{marker_shape};
                        transform:{"none" if in_bld else "rotate(-45deg)"};
                        border:2px solid white;
                        box-shadow:2px 2px 6px rgba(0,0,0,0.5);
                        display:flex;align-items:center;justify-content:center;
                    '><span style='transform:{"none" if in_bld else "rotate(45deg)"}'>{rk}</span></div>""",
                    icon_size=(30, 30),
                    icon_anchor=(15, 15 if in_bld else 30),
                ),
            ).add_to(fg_cand)

        fg_cand.add_to(m)

    # ── Plugins ───────────────────────────────────────────────────────────────
    plugins.Fullscreen(position="topleft", title="Pantalla completa").add_to(m)
    plugins.MiniMap(toggle_display=True, tile_layer="OpenStreetMap",
                    width=130, height=130).add_to(m)
    plugins.MousePosition(
        position="bottomleft",
        separator=" | ",
        empty_string="—",
        lat_formatter="function(num) {return L.Util.formatNum(num, 6) + '° N';}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 6) + '° O';}",
    ).add_to(m)

    # ── Layer control ─────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # ── Title overlay ─────────────────────────────────────────────────────────
    title_html = """
    <div style="
        position:fixed;top:12px;left:50%;transform:translateX(-50%);
        z-index:1000;background:rgba(15,30,55,0.92);
        padding:10px 22px;border-radius:10px;
        font-family:'Segoe UI',sans-serif;color:white;
        box-shadow:0 4px 14px rgba(0,0,0,0.45);
        border:1px solid rgba(255,255,255,0.15);
        backdrop-filter:blur(4px);pointer-events:none;
        text-align:center;
    ">
      <div style='font-size:15px;font-weight:bold;letter-spacing:0.3px'>
        🎓 Yachay Tech — Estaciones de Recarga de Agua
      </div>
      <div style='font-size:11px;opacity:0.75;margin-top:3px'>
        Urcuquí, Imbabura, Ecuador &nbsp;|&nbsp; Análisis multi-criterio
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items_html = ""
    for z in ZONES.values():
        legend_items_html += f"""
        <div style='display:flex;align-items:center;gap:6px;margin:4px 0'>
          <div style='width:14px;height:14px;border-radius:50%;
                      background:{z["light"]};border:2.5px solid {z["color"]}'></div>
          <span>{z["name"]}</span>
        </div>"""
    legend_items_html += """
        <hr style='margin:6px 0;border-color:#ddd'>
        <div style='display:flex;align-items:center;gap:6px;margin:4px 0'>
          <div style='width:14px;height:14px;border-radius:4px;
                      background:#1b5e20;border:2px solid white;
                      box-shadow:1px 1px 3px rgba(0,0,0,0.4)'></div>
          <span>Candidato dentro de edificio</span>
        </div>
        <div style='display:flex;align-items:center;gap:6px;margin:4px 0'>
          <div style='width:14px;height:14px;border-radius:50% 50% 50% 0;
                      transform:rotate(-45deg);background:#c62828;
                      border:2px solid white;box-shadow:1px 1px 3px rgba(0,0,0,0.4)'></div>
          <span>Candidato adyacente a edificio</span>
        </div>"""

    legend_html = f"""
    <div style="
        position:fixed;bottom:30px;right:10px;z-index:1000;
        background:rgba(255,255,255,0.96);padding:12px 15px;
        border-radius:10px;box-shadow:0 3px 12px rgba(0,0,0,0.3);
        font-family:'Segoe UI',sans-serif;font-size:12px;
        border:1px solid #ddd;min-width:195px;
    ">
      <div style='font-weight:bold;font-size:13px;margin-bottom:8px;color:#1a3a5c'>
        📌 Leyenda
      </div>
      {legend_items_html}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    out = OUT_I / "yachay_tech_water_stations.html"
    m.save(str(out))
    print(f"   Interactive map saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Multi-page PDF Report
# ══════════════════════════════════════════════════════════════════════════════
def export_report():
    print("── Assembling PDF report …")
    map_files = sorted(OUT.glob("*.png"))
    if not map_files:
        print("   No PNG maps found — run scripts 02-04 first.")
        return

    pdf_path = BASE / "outputs" / "Yachay_Tech_Water_Stations_Report.pdf"

    captions = {
        "01_terrain_3d.png":
            "Figura 1 — Modelo Digital de Elevación 3D del campus de Yachay Tech y sus alrededores (SRTM 30 m). "
            "Los tres sitios de estudio están marcados con puntos de color.",
        "02_contour_hillshade.png":
            "Figura 2 — Mapa topográfico con curvas de nivel (intervalo 10 m), sombras del terreno, "
            "edificaciones y red vial del campus.",
        "03_shade_analysis.png":
            "Figura 3 — Análisis de sombra: sombra del terreno, proyección de sombras de edificaciones "
            "y el índice combinado utilizado en la ponderación de candidatos.",
        "04_solar_annual.png":
            "Figura 4 — Mapa de radiación solar anual promedio (GHI, kWh/m²/día) con tabla mensual. "
            "Datos: NASA POWER Climatology API.",
        "05_solar_monthly.png":
            "Figura 5 — Panel de 12 mapas con la variación mensual de la irradiancia horizontal global (GHI). "
            "El mes con mayor GHI se resalta en rojo; el de menor GHI en azul.",
        "06_scoring_candidates.png":
            "Figura 6 — Análisis multi-criterio: capas de puntuación solar, sombra y proximidad; "
            "mapa combinado con los 8 candidatos prioritarios y su tabla de ubicación.",
    }

    with PdfPages(str(pdf_path)) as pdf:
        # ── Cover ──────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        fig.patch.set_facecolor("#0d1f35")

        # Gradient-ish background
        bg = np.linspace(0, 1, 200).reshape(1, -1)
        ax.imshow(bg, aspect="auto", cmap="Blues", alpha=0.18,
                  extent=[0, 1, 0, 1], transform=ax.transAxes)

        ax.text(0.5, 0.83, "YACHAY TECH UNIVERSITY",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=22, fontweight="bold", color="white",
                fontfamily="DejaVu Sans")
        ax.text(0.5, 0.73,
                "Estudio de Ubicación para Estaciones\nde Recarga de Agua",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=18, color="#81d4fa", linespacing=1.5)
        ax.text(0.5, 0.62, "Urcuquí · Imbabura · Ecuador",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#90caf9", style="italic")

        # Zone summary boxes
        zone_list = list(ZONES.values())
        box_positions = [0.22, 0.50, 0.78]
        for zv, bx in zip(zone_list, box_positions):
            rect = mpatches.FancyBboxPatch((bx - 0.12, 0.41), 0.24, 0.12,
                                            boxstyle="round,pad=0.01",
                                            facecolor=zv["color"], alpha=0.85,
                                            transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.text(bx, 0.495, zv["name"], ha="center", va="center",
                    transform=ax.transAxes, fontsize=9.5,
                    fontweight="bold", color="white")

        method_lines = (
            "Metodología:\n"
            "  • Datos geoespaciales del campus: OpenStreetMap (osmnx)\n"
            "  • Elevación / topografía: SRTM 30 m (NASA)\n"
            "  • Radiación solar (GHI): NASA POWER Climatology API\n"
            "  • Análisis de sombra: hillshade del terreno + proyección de edificaciones\n"
            "  • Ponderación: Radiación solar 30% · Sombra 20% · Edificaciones 35% · Proximidad 15%\n"
            "  • Herramientas: Python · geopandas · rasterio · matplotlib · folium"
        )
        ax.text(0.5, 0.24, method_lines,
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#cfd8dc", family="monospace", linespacing=1.55,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#0a1929", alpha=0.75,
                          edgecolor="#1e3a5f"))

        ax.text(0.5, 0.06,
                "Análisis generado con datos públicos de libre acceso",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#546e7a", style="italic")

        pdf.savefig(fig, facecolor="#0d1f35")
        plt.close(fig)

        # ── Map pages ──────────────────────────────────────────────────────────
        for img_path in map_files:
            fig = plt.figure(figsize=(11, 8.5), facecolor="white")
            ax  = fig.add_axes([0.02, 0.09, 0.96, 0.87])
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.set_axis_off()

            caption = captions.get(img_path.name, img_path.stem.replace("_", " ").title())
            fig.text(0.5, 0.035, caption, ha="center", va="bottom",
                     fontsize=7.5, style="italic", color="#444", wrap=True)
            fig.text(0.98, 0.01, "Yachay Tech — Water Station Analysis",
                     ha="right", va="bottom", fontsize=6.5, color="#aaa")
            fig.text(0.02, 0.01, "Fuente: OpenStreetMap, NASA SRTM, NASA POWER",
                     ha="left", va="bottom", fontsize=6.5, color="#aaa")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Solar data table ───────────────────────────────────────────────────
        df_solar = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
        months_es = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                     "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre","Anual"]

        fig, ax = plt.subplots(figsize=(11, 4), facecolor="white")
        ax.axis("off")
        fig.patch.set_facecolor("white")

        tbl_data = []
        for i, (idx_m, row) in enumerate(df_solar.iterrows()):
            tbl_data.append([
                months_es[i],
                f"{row['ALLSKY_SFC_SW_DWN']:.4f}",
                f"{row['CLRSKY_SFC_SW_DWN']:.4f}",
                f"{row['T2M']:.2f}",
            ])

        col_labels = ["Mes", "GHI — Cielo Total\n(kWh/m²/día)",
                       "GHI — Cielo Despejado\n(kWh/m²/día)",
                       "Temperatura Media\n(°C)"]
        tbl = ax.table(cellText=tbl_data, colLabels=col_labels,
                        cellLoc="center", loc="center",
                        bbox=[0.0, 0.0, 1.0, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        ghi_vals = [float(df_solar.iloc[i]["ALLSKY_SFC_SW_DWN"]) for i in range(12)]
        max_ghi, min_ghi = max(ghi_vals), min(ghi_vals)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_facecolor("#1a3a5c")
                cell.set_text_props(color="white", fontweight="bold")
            elif r == len(tbl_data):   # Annual row
                cell.set_facecolor("#fff9c4")
                cell.set_text_props(fontweight="bold")
            elif r <= 12:
                val = ghi_vals[r - 1]
                if val == max_ghi:
                    cell.set_facecolor("#ffcdd2")
                elif val == min_ghi:
                    cell.set_facecolor("#bbdefb")
                elif r % 2 == 0:
                    cell.set_facecolor("#f5f5f5")

        ax.set_title("Tabla 2 — Datos de Radiación Solar NASA POWER (Climatología)\n"
                     "Parámetro: ALLSKY_SFC_SW_DWN  |  Coordenadas: 0.4052°N, 78.1760°O  |  "
                     "Altitud: ~2400 m s.n.m.",
                     fontsize=10, fontweight="bold", pad=12)
        fig.text(0.5, 0.01,
                 "Rojo = Mes de mayor GHI  |  Azul = Mes de menor GHI  |  Amarillo = Promedio anual",
                 ha="center", fontsize=8, color="#555")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Candidate table ────────────────────────────────────────────────────
        cand_path = PROC / "candidates.csv"
        if cand_path.exists():
            dfc = pd.read_csv(cand_path)
            tr  = Transformer.from_crs(EPSG_LOCAL, 4326, always_xy=True)
            lons, lats = tr.transform(dfc.easting.values, dfc.northing.values)

            fig, ax = plt.subplots(figsize=(13, 3.8), facecolor="white")
            ax.axis("off")
            tbl_data = [[
                f"#{int(r['Rango'])}",
                str(r.get("anchor_label", "Sitio")),
                f"{lats[i]:.6f}",
                f"{lons[i]:.6f}",
                f"{r['score']:.4f}",
                "SI" if r.get("in_building", False) else "NO",
                str(r.get("building_name", "—"))[:25],
                str(r.get("building_type", "—")),
                min(ZONES.values(),
                    key=lambda z: (z["lat"]-lats[i])**2+(z["lon"]-lons[i])**2)["name"],
                str(int(r.get("cluster_size", 0))),
            ] for i, (_, r) in enumerate(dfc.iterrows())]

            col_labels = ["Rango","Sitio Requerido","Latitud (deg)","Longitud (deg)",
                           "Puntaje","En Edificio","Nombre Edificio",
                           "Tipo","Zona Mas Cercana","Cluster\nPixeles"]
            tbl = ax.table(cellText=tbl_data, colLabels=col_labels,
                            cellLoc="center", loc="center",
                            bbox=[0.0, 0.0, 1.0, 1.0])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            rank_colors = ["#c62828","#e65100","#ef6c00","#1b5e20",
                           "#006064","#4a148c","#880e4f","#37474f"]
            for (r2, c2), cell in tbl.get_celld().items():
                cell.set_edgecolor("#cccccc")
                if r2 == 0:
                    cell.set_facecolor("#1a3a5c")
                    cell.set_text_props(color="white", fontweight="bold")
                elif r2 <= len(tbl_data):
                    if c2 == 0:
                        cell.set_facecolor(rank_colors[(r2-1) % len(rank_colors)])
                        cell.set_text_props(color="white", fontweight="bold")
                    elif c2 == 5:
                        val = tbl_data[r2-1][5]
                        cell.set_facecolor("#c8e6c9" if val == "SI" else "#ffcdd2")
                        cell.set_text_props(fontweight="bold")
                    elif r2 % 2 == 0:
                        cell.set_facecolor("#f5f5f5")
            ax.set_title(
                "Tabla 3 — Candidatos Restringidos para Estaciones de Recarga de Agua\n"
                f"Solar {int(W_SOLAR*100)}% · Sombra {int(W_SHADE*100)}% · "
                f"Edificaciones {int(W_BUILDING*100)}% · Proximidad {int(W_PROXIMITY*100)}%  "
                "| Verde = dentro del edificio  |  Rojo = adyacente",
                fontsize=9.5, fontweight="bold", pad=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"   Report saved: {pdf_path}")


if __name__ == "__main__":
    export_interactive()
    export_geospatial_data_science_outputs()
    export_report()
    print("\n✓ All exports complete.")
