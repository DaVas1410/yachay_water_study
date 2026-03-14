# Yachay Tech Water Refill Station Analysis

Geospatial analysis project to identify optimal locations for water refilling stations at **Yachay Tech University (Urcuquí, Ecuador)**, including:

- Campus topography and terrain visualizations
- Solar radiation heatmaps (annual + monthly)
- Shade and accessibility scoring
- Constrained candidate selection using stakeholder-required sites
- Interactive web map + report outputs

## Study zones

- Campus Principal
- Jardín Botánico
- Innopolis / Centro de Emprendimiento

## Main outputs

- `outputs/maps/`: static figures (`.png`, `.pdf`)
- `outputs/interactive/yachay_tech_water_stations.html`: interactive map
- `outputs/Yachay_Tech_Water_Stations_Report.pdf`: report PDF
- `outputs/report_data/`: geospatial data-science exports (`.csv`, `.geojson`, `.json`)

## Project structure

```text
agua_csu/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── maps/
│   ├── interactive/
│   └── report_data/
├── scripts/
│   ├── config.py
│   ├── 01_fetch_data.py
│   ├── 02_terrain_plots.py
│   ├── 03_solar_heatmaps.py
│   ├── 04_scoring.py
│   └── 05_export_report.py
└── requirements.txt
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run pipeline

```bash
python3 scripts/01_fetch_data.py
python3 scripts/02_terrain_plots.py
python3 scripts/03_solar_heatmaps.py
python3 scripts/04_scoring.py
python3 scripts/06_intuitive_figures.py
python3 scripts/05_export_report.py
```

## Notes

- Solar data source: NASA POWER (`ALLSKY_SFC_SW_DWN`)
- Coordinate system for analysis: **EPSG:32717 (UTM 17S)**
- Candidate selection is constrained to stakeholder-priority anchors and building feasibility.
