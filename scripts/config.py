"""
Shared configuration for all Yachay Tech water station analysis scripts.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
OUT  = BASE / "outputs" / "maps"
OUT_I = BASE / "outputs" / "interactive"

# Study area
STUDY_CENTER_LAT = 0.4125
STUDY_CENTER_LON = -78.1846
EPSG_LOCAL = 32717  # UTM Zone 17S

# ── Campus zones ────────────────────────────────────────────────────────────
ZONES = {
    "main": {
        "name": "Campus Principal",
        "lat": 0.4052,
        "lon": -78.1760,
        "color": "#1565C0",   # deep blue
        "light": "#BBDEFB",
        "marker": "●",
        "radius_m": 900,
    },
    "botanico": {
        "name": "Jardín Botánico",
        "lat": 0.4183386970866529,
        "lon": -78.18790548627916,
        "color": "#2E7D32",   # deep green
        "light": "#C8E6C9",
        "marker": "●",
        "radius_m": 400,
    },
    "innopolis": {
        "name": "Innopolis / C. Emprendimiento",
        "lat": 0.41984081733076445,
        "lon": -78.18990836679563,
        "color": "#E65100",   # deep orange
        "light": "#FFE0B2",
        "marker": "●",
        "radius_m": 350,
    },
}

# Scoring weights (must sum to 1.0)
W_SOLAR     = 0.30   # high solar → high water demand
W_SHADE     = 0.20   # shaded → more comfortable placement
W_BUILDING  = 0.35   # inside/near buildings → physically installable
W_PROXIMITY = 0.15   # near pedestrian paths → foot traffic
TOP_N       = 8

# DBSCAN clustering parameters
CLUSTER_EPS_M   = 200   # metres: max distance between core points in same cluster
CLUSTER_MIN_PTS = 5     # minimum pixels to form a cluster

# Sun angles for Urcuquí at solar noon
SUN_AZIMUTH  = 180
SUN_ALTITUDE = 65

# Style defaults
TITLE_FONT   = {"fontsize": 14, "fontweight": "bold", "pad": 14}
LABEL_FONT   = {"fontsize": 9}
LEGEND_FONT  = 9
FIG_DPI      = 200
