#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KEGG und GO(BP) Enrichment für Dabrafenib-Gene mit gseapy.

Ausgabe:
- dabrafenib_kegg_enrichment.csv
- dabrafenib_go_bp_enrichment.csv
- Ordner: dabrafenib_kegg/, dabrafenib_go_bp/ mit Plots (von gseapy)
"""

import os
import pandas as pd
import gseapy as gp


def main():
    # -----------------------------------------------------
    # 1. Gene-Liste ohne Präfixe (deine alte + neue Gene)
    # -----------------------------------------------------
    genes = [
        # --- neue Gene aus deiner aktuellen Liste ---
        "IFNWP19", "NEU1", "FAM86C", "NRIP1", "ABCD3", "PPP6C",
        "ACTN1", "NFATC2", "PCCA", "RAPH1", "RBM38", "YIPF3",
        "HMGCR", "PTK6", "C14orf153", "LINC02315", "MIR320A",
    ]

    # Duplikate entfernen & sortieren
    genes = sorted(set(genes))

    print(f"{len(genes)} Gene für Enrichment:\n{genes}\n")

    # -----------------------------------------------------
    # 2. KEGG Enrichment
    # -----------------------------------------------------
    print("Starte KEGG-Enrichment ...")
    kegg_res = gp.enrichr(
        gene_list=genes,
        gene_sets=["KEGG_2019_Human"],
        organism="Human",
        outdir="dabrafenib_kegg",
        cutoff=0.5,
        no_plot=False,
    )

    if hasattr(kegg_res, "results") and not kegg_res.results.empty:
        kegg_df = kegg_res.results
        kegg_df.to_csv("dabrafenib_kegg_enrichment.csv", index=False)
        print("KEGG-Ergebnisse gespeichert: dabrafenib_kegg_enrichment.csv")
    else:
        print("KEGG: Keine Ergebnisse erhalten.")

    # -----------------------------------------------------
    # 3. GO Biological Process Enrichment
    # -----------------------------------------------------
    print("\nStarte GO BP-Enrichment ...")
    go_res = gp.enrichr(
        gene_list=genes,
        gene_sets=["GO_Biological_Process_2021"],
        organism="Human",
        outdir="dabrafenib_go_bp",
        cutoff=0.5,
        no_plot=False,
    )

    if hasattr(go_res, "results") and not go_res.results.empty:
        go_df = go_res.results
        go_df.to_csv("dabrafenib_go_bp_enrichment.csv", index=False)
        print("GO BP-Ergebnisse gespeichert: dabrafenib_go_bp_enrichment.csv")
    else:
        print("GO BP: Keine Ergebnisse erhalten.")

    print("\nFertig. Siehe Ordner: dabrafenib_kegg/ und dabrafenib_go_bp/.")


if __name__ == "__main__":
    main()
