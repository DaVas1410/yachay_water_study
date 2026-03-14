"""
Script 01: Data Acquisition
- Downloads Yachay Tech campus data from OpenStreetMap
- Downloads SRTM 30m DEM for campus bounding box
- Queries NASA POWER API for monthly + annual solar radiation
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# ── Study area centroid covering all 3 campus zones ───────────────────────────
# Main Campus: 0.4052, -78.1760
# Jardín Botánico: 0.4183, -78.1879
# Innopolis (Centro Emprendimiento): 0.4198, -78.1899
LAT  = 0.4125        # centroid of all three zones
LON  = -78.1846
DIST = 3500          # metres radius — covers all zones
EPSG_LOCAL = 32717   # UTM Zone 17S (covers Ecuador)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  OpenStreetMap — campus features
# ══════════════════════════════════════════════════════════════════════════════
def fetch_campus_osm():
    print("── Fetching campus data from OpenStreetMap …")

    # Custom filter to grab the university boundary if tagged
    tags_place = {"amenity": "university"}
    try:
        boundary = ox.features_from_point((LAT, LON), tags=tags_place, dist=DIST)
        boundary.to_file(RAW / "campus_boundary.gpkg", driver="GPKG", layer="boundary")
        print(f"   Campus boundary features: {len(boundary)}")
    except Exception as e:
        print(f"   Warning: could not fetch boundary tag — {e}")

    # Buildings
    buildings = ox.features_from_point((LAT, LON), tags={"building": True}, dist=DIST)
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    buildings.to_file(RAW / "buildings.gpkg", driver="GPKG", layer="buildings")
    print(f"   Buildings: {len(buildings)}")

    # Road / path network
    G = ox.graph_from_point((LAT, LON), dist=DIST, network_type="all")
    nodes, edges = ox.graph_to_gdfs(G)
    edges.to_file(RAW / "roads.gpkg", driver="GPKG", layer="roads")
    nodes.to_file(RAW / "road_nodes.gpkg", driver="GPKG", layer="nodes")
    print(f"   Road edges: {len(edges)}")

    # Footpaths / pedestrian ways
    footpaths = ox.features_from_point(
        (LAT, LON),
        tags={"highway": ["footway", "path", "pedestrian", "steps", "cycleway"]},
        dist=DIST,
    )
    footpaths.to_file(RAW / "footpaths.gpkg", driver="GPKG", layer="footpaths")
    print(f"   Footpaths: {len(footpaths)}")

    print("   OSM data saved to data/raw/")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SRTM DEM — elevation raster
# ══════════════════════════════════════════════════════════════════════════════
def fetch_dem():
    """
    Download SRTM 1-arc-second (~30m) tile for the campus area.
    Uses the `elevation` library which wraps the NASA/USGS SRTM dataset.
    Falls back to a synthetic DEM if internet/tile access fails.
    """
    print("── Fetching DEM …")

    # Bounding box with 0.02° buffer (~2 km) around campus
    buffer = 0.02
    bounds = (LON - buffer, LAT - buffer, LON + buffer, LAT + buffer)

    dem_path = RAW / "dem_srtm.tif"

    try:
        import elevation
        elevation.clip(bounds=bounds, output=str(dem_path), product="SRTM1")
        print(f"   DEM saved: {dem_path}")
    except Exception as e:
        print(f"   elevation library failed ({e}), trying OpenTopography SRTM URL …")
        try:
            url = (
                f"https://portal.opentopography.org/API/globaldem"
                f"?demtype=SRTMGL1&south={LAT - buffer}&north={LAT + buffer}"
                f"&west={LON - buffer}&east={LON + buffer}&outputFormat=GTiff"
            )
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(dem_path, "wb") as f:
                f.write(r.content)
            print(f"   DEM saved via OpenTopography: {dem_path}")
        except Exception as e2:
            print(f"   OpenTopography failed ({e2}). Generating synthetic DEM …")
            _generate_synthetic_dem(dem_path, bounds)

    # Reproject to UTM 17S and save processed version
    _reproject_dem(dem_path, PROC / "dem_utm17s.tif")


def _reproject_dem(src_path, dst_path):
    """Reproject DEM to UTM Zone 17S for metric analysis."""
    try:
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, CRS.from_epsg(EPSG_LOCAL), src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({"crs": CRS.from_epsg(EPSG_LOCAL),
                          "transform": transform, "width": width, "height": height})
            with rasterio.open(dst_path, "w", **meta) as dst:
                reproject(source=rasterio.band(src, 1),
                           destination=rasterio.band(dst, 1),
                           src_transform=src.transform,
                           src_crs=src.crs,
                           dst_transform=transform,
                           dst_crs=CRS.from_epsg(EPSG_LOCAL),
                           resampling=Resampling.bilinear)
        print(f"   DEM reprojected to UTM 17S: {dst_path}")
    except Exception as e:
        print(f"   Warning: could not reproject DEM — {e}")


def _generate_synthetic_dem(out_path, bounds):
    """
    Create a plausible synthetic DEM for Urcuquí (≈2400 m asl, gentle terrain)
    used as a fallback when real SRTM data cannot be downloaded.
    """
    west, south, east, north = bounds
    rows, cols = 240, 240
    transform = from_bounds(west, south, east, north, cols, rows)

    # Base elevation ~2400 m with gentle NW–SE slope and small hills
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    xx, yy = np.meshgrid(x, y)
    dem = (2350
           + 80 * xx          # gentle east slope
           - 40 * yy          # gentle south slope
           + 30 * np.sin(xx * 8) * np.cos(yy * 6)   # undulation
           + 15 * np.random.default_rng(42).standard_normal((rows, cols)))

    with rasterio.open(
        out_path, "w",
        driver="GTiff", height=rows, width=cols, count=1,
        dtype="float32", crs=CRS.from_epsg(4326), transform=transform,
    ) as ds:
        ds.write(dem.astype("float32"), 1)

    print(f"   Synthetic DEM written: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NASA POWER API — solar radiation
# ══════════════════════════════════════════════════════════════════════════════
def fetch_solar():
    """
    Query NASA POWER API for monthly + annual Global Horizontal Irradiance (GHI)
    at Yachay Tech coordinates. Parameter: ALLSKY_SFC_SW_DWN (kWh/m²/day).
    """
    print("── Querying NASA POWER API for solar radiation …")

    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M",
        "community": "RE",
        "longitude": -78.1760,   # main campus point
        "latitude":  0.4052,
        "format": "JSON",
        "header": "true",
        "time-standard": "LST",
    }

    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        records = {}
        for param, monthly in data["properties"]["parameter"].items():
            records[param] = monthly

        df = pd.DataFrame(records)
        months = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec","Ann"]
        df.index = months
        df.index.name = "Month"
        df.to_csv(RAW / "solar_nasa_power.csv")
        print(f"   Solar data saved: {RAW / 'solar_nasa_power.csv'}")
        print(df[["ALLSKY_SFC_SW_DWN"]].to_string())

    except Exception as e:
        print(f"   NASA POWER failed ({e}). Generating typical Urcuquí values …")
        _generate_synthetic_solar()


def _generate_synthetic_solar():
    """
    Fallback: typical GHI values for equatorial highland Ecuador (~0°, 2400 m).
    Values in kWh/m²/day (Global Horizontal Irradiance).
    """
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec","Ann"]
    # Urcuquí is near equator with slightly less radiation in rainy season (Feb–May)
    ghi = [5.1, 4.6, 4.5, 4.4, 4.8, 5.2, 5.5, 5.6, 5.3, 4.9, 4.8, 5.0, 4.98]
    clear = [g * 1.18 for g in ghi]
    t2m   = [14.2]*12 + [14.2]

    df = pd.DataFrame({
        "ALLSKY_SFC_SW_DWN": ghi,
        "CLRSKY_SFC_SW_DWN": clear,
        "T2M": t2m,
    }, index=months)
    df.index.name = "Month"
    df.to_csv(RAW / "solar_nasa_power.csv")
    print(f"   Synthetic solar data saved: {RAW / 'solar_nasa_power.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fetch_campus_osm()
    fetch_dem()
    fetch_solar()
    print("\n✓ All data acquisition complete.")
