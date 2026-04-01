"""
UV Index Analysis - Clean visualizations without annotations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT.mkdir(parents=True, exist_ok=True)


def _uv_annual_bar():
    """Annual UV Index by month - simple bar chart."""
    df = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
    months_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    
    uv_vals = [float(df.loc[m, "UV_INDEX"]) for m in df.index[:-1]]  # exclude Ann
    annual_uv = float(df.loc["Ann", "UV_INDEX"])
    
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    colors = ["#ef6c00" if v > annual_uv else "#1e88e5" for v in uv_vals]
    ax.bar(months_es, uv_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.axhline(annual_uv, color="#424242", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Índice UV", fontsize=11, fontweight="bold")
    ax.set_title("Índice UV Mensual", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    
    fig.tight_layout()
    fig.savefig(OUT / "14_uv_index_monthly.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "14_uv_index_monthly.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ UV monthly bar chart created")


def _uv_trend():
    """UV Index trend line."""
    df = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
    months_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    
    uv_vals = [float(df.loc[m, "UV_INDEX"]) for m in df.index[:-1]]
    annual_uv = float(df.loc["Ann", "UV_INDEX"])
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.plot(months_es, uv_vals, marker="o", color="#e53935", linewidth=2.5, markersize=7,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.axhline(annual_uv, color="#424242", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.fill_between(range(len(months_es)), uv_vals, alpha=0.2, color="#e53935")
    
    ax.set_ylabel("Índice UV", fontsize=11, fontweight="bold")
    ax.set_title("Tendencia de Índice UV", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    
    fig.tight_layout()
    fig.savefig(OUT / "15_uv_index_trend.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "15_uv_index_trend.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ UV trend line created")


def _solar_uv_comparison():
    """Side-by-side comparison of Solar vs UV Index."""
    df = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
    months_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    
    solar_vals = np.array([float(df.loc[m, "ALLSKY_SFC_SW_DWN"]) for m in df.index[:-1]])
    uv_vals = np.array([float(df.loc[m, "UV_INDEX"]) for m in df.index[:-1]])
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    
    # Solar
    axes[0].bar(months_es, solar_vals, color="#ffa726", alpha=0.85, edgecolor="black", linewidth=0.8)
    axes[0].set_ylabel("kWh/m²/día", fontsize=10, fontweight="bold")
    axes[0].set_title("Radiación Solar", fontsize=11, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[0].tick_params(axis="x", rotation=45)
    
    # UV
    axes[1].bar(months_es, uv_vals, color="#e53935", alpha=0.85, edgecolor="black", linewidth=0.8)
    axes[1].set_ylabel("Índice UV", fontsize=10, fontweight="bold")
    axes[1].set_title("Índice UV", fontsize=11, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[1].tick_params(axis="x", rotation=45)
    
    fig.suptitle("Comparación: Radiación Solar vs Índice UV", fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "16_solar_uv_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "16_solar_uv_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Solar vs UV comparison created")


def _uv_distribution():
    """UV Index distribution across year."""
    df = pd.read_csv(RAW / "solar_nasa_power.csv", index_col=0)
    
    uv_vals = [float(df.loc[m, "UV_INDEX"]) for m in df.index[:-1]]
    
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.hist(uv_vals, bins=6, color="#e53935", alpha=0.7, edgecolor="black", linewidth=1.2)
    ax.axvline(np.mean(uv_vals), color="#424242", linestyle="--", linewidth=2, label=f"Promedio: {np.mean(uv_vals):.2f}")
    
    ax.set_xlabel("Índice UV", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frecuencia (meses)", fontsize=11, fontweight="bold")
    ax.set_title("Distribución del Índice UV", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT / "17_uv_distribution.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "17_uv_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ UV distribution histogram created")


if __name__ == "__main__":
    print("☀️ Generating UV Index analysis...")
    _uv_annual_bar()
    _uv_trend()
    _solar_uv_comparison()
    _uv_distribution()
    print("✅ UV Index analysis complete!")
