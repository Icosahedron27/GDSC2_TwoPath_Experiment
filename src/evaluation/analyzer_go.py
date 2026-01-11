import os
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib muss in einen beschreibbaren Cache schreiben; setze lokalen Pfad bevor es importiert wird.
MPL_CACHE = Path("tmp/mplcache")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE.resolve()))

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Einstellungen
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "docetaxel_kegg_enrichment.csv"   # Pfad zur CSV im gleichen Ordner
OUT_DIR = SCRIPT_DIR / "enrichment_plots"
OUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Einlesen
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

# Spalten-Namen prüfen (je nach gseapy-Version leicht anders)
print(df.columns)

# -------------------------------------------------------------------
# Overlap-String ("2/124") in Zahlen umwandeln
# -------------------------------------------------------------------
overlap_split = df["Overlap"].str.split("/", expand=True)
df["k_in_path"] = overlap_split[0].astype(int)    # Anzahl deiner Gene im Pfad
df["M_in_path"] = overlap_split[1].astype(int)    # Anzahl Gene im Pfad insgesamt
df["gene_ratio"] = df["k_in_path"] / df["M_in_path"]

# zusätzliche Kennzahl: -log10(p)
df["minus_log10_p"] = -np.log10(df["P-value"])

# -------------------------------------------------------------------
# Filter: "interessante" Pfade (z.B. FDR < 0.25, explorativ)
# -------------------------------------------------------------------
filt = df[df["Adjusted P-value"] < 0.25].copy()

# Falls nichts durchkommt: fallback auf Top-N nach Combined Score
if filt.empty:
    filt = df.copy()

# sortieren nach Combined Score (oder minus_log10_p)
filt = filt.sort_values("Combined Score", ascending=False)

# -------------------------------------------------------------------
# Top-N auswählen
# -------------------------------------------------------------------
TOP_N = 20
top = filt.head(TOP_N).copy()

# Term für Plot etwas kürzen, damit es in die Achse passt
top["Term_short"] = top["Term"].str.replace(" (human)", "", regex=False)
top["Term_short"] = top["Term_short"].str.slice(0, 60)

# -------------------------------------------------------------------
# 1) Horizontaler Barplot: Combined Score
# -------------------------------------------------------------------
plt.figure(figsize=(8, 0.4 * TOP_N + 2))

y_pos = np.arange(len(top))[::-1]  # damit Top-Term oben steht

plt.barh(y_pos, top["Combined Score"])
plt.yticks(y_pos, top["Term_short"])
plt.xlabel("Combined Score")
plt.title("Top KEGG/GO Terms (Combined Score)")

plt.tight_layout()
plt.savefig(OUT_DIR / "docetaxel_top_terms_combined_score.png", dpi=300)
plt.close()

# -------------------------------------------------------------------
# 2) Dot-Plot: Gene Ratio vs. -log10(p)
# -------------------------------------------------------------------
plt.figure(figsize=(8, 0.4 * TOP_N + 2))

y_pos = np.arange(len(top))[::-1]

# Punktgröße z.B. proportional zur Anzahl deiner Gene im Pfad
sizes = 50 + 50 * (top["k_in_path"] - top["k_in_path"].min()) / max(
    1, (top["k_in_path"].max() - top["k_in_path"].min())
)

plt.scatter(top["gene_ratio"], y_pos, s=sizes)
plt.yticks(y_pos, top["Term_short"])
plt.xlabel("Gene Ratio (k_in_path / M_in_path)")
plt.title("Top KEGG/GO Terms: Gene Ratio vs. -log10(p)")
# Zweite X-Achse für -log10(p) optional:
# oder alternativ die x-Achse mit minus_log10_p und gene_ratio in Farbe kodieren

plt.tight_layout()
plt.savefig(OUT_DIR / "docetaxel_top_terms_gene_ratio.png", dpi=300)
plt.close()

print("Plots gespeichert in", OUT_DIR.resolve())
