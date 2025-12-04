# Biomarker Korrelationsanalyse & Proxy-Gruppenbildung

## Überblick

Diese Analyse untersucht die Korrelation zwischen häufig vorkommenden Biomarkern (frequent itemsets) 
aus den CPSS-Ergebnissen beider Pfade (RF und Linear) und bildet Proxy-Gruppen basierend auf hoher Korrelation.

**Methodologie:**
- **Korrelationsberechnung**: Pearson-Korrelation über 100 zufällig ausgewählte Medikamente
- **Clustering**: Hierarchisches Clustering mit Complete-Linkage
- **Gruppenschwelle**: r ≥ 0.7 (hohe Korrelation)
- **Minimum Support**: 15 Medikamente pro Itemset

---

## RF-Pfad (Non-Linear)

### Statistik
- **Total Biomarker**: 21
- **Proxy-Gruppen**: 7
- **Durchschnittliche Gruppengröße**: 3.0
- **Größte Gruppe**: 8 Mitglieder
- **Singletons**: 2

### Top Proxy-Gruppen

#### 1. Methylierungs-Cluster (n=8, r̄=0.80)
**Representative**: `METH__TMEM51`

**Mitglieder**:
- METH__C9orf150
- METH__DSTN
- METH__KLF4
- METH__LAMB2
- METH__PTPN14
- METH__TEAD4
- METH__TMEM51 ⭐
- METH__ZNF697

**Intragruppen-Korrelation**: 0.72 - 0.86 (avg: 0.80)

**Interpretation**: Diese Methylierungsmarker sind stark ko-reguliert und können als funktionelle Einheit 
betrachtet werden. `TMEM51` hat die zentralste Position (höchste durchschnittliche Korrelation mit allen 
Gruppenmitgliedern) und eignet sich als Proxy für die gesamte Gruppe.

---

#### 2. Expression-Cluster (n=4, r̄=0.82)
**Representative**: `TPM__NCKAP1L`

**Mitglieder**:
- TPM__ARHGAP15
- TPM__IKZF1
- TPM__NCKAP1L ⭐
- TPM__PDE1B

**Intragruppen-Korrelation**: 0.75 - 0.91 (avg: 0.82)

**Top Paarung**: `NCKAP1L` ↔ `PDE1B` (r=0.91)

**Interpretation**: Expression-Signaturen mit sehr starker Ko-Expression. `NCKAP1L` ist der optimalste 
Vertreter dieser Gruppe.

---

#### 3. Methylierungs-Expression Hybrid (n=3, r̄=0.77)
**Representative**: `METH__CCM2`

**Mitglieder**:
- METH__CCM2 ⭐
- METH__PRIC285
- TPM__TJP1

**Intragruppen-Korrelation**: 0.72 - 0.80 (avg: 0.77)

**Interpretation**: Interessante Cross-Omic Gruppe, die Methylierung und Expression vereint.

---

#### 4. Expression Pair 1 (n=2, r=0.81)
- TPM__PIK3CG ⭐
- TPM__RCSD1

---

#### 5. Expression Pair 2 (n=2, r=0.81)
- TPM__SDC4 ⭐
- TPM__TNFRSF12A

---

#### 6-7. Singletons
- TPM__MIR142HG (keine starke Korrelation mit anderen Markern)
- TPM__PTPN7 (unabhängiger Marker)

---

## Linear-Pfad (ElasticNet)

### Statistik
- **Total Biomarker**: 4
- **Proxy-Gruppen**: 4
- **Durchschnittliche Gruppengröße**: 1.0
- **Größte Gruppe**: 1 Mitglied
- **Singletons**: 4

### Marker (alle unabhängig)

1. **TPM__APLP2** - Unabhängiger Expression-Marker
2. **TPM__BCL2L1** - Unabhängiger Expression-Marker (höchster Support: 94 drugs, 36.7%)
3. **TPM__MOB3A** - Unabhängiger Expression-Marker
4. **TPM__SLC38A5** - Unabhängiger Expression-Marker

**Interpretation**: Der lineare Pfad identifiziert wenige, aber hochgradig unabhängige Biomarker. 
Keine Korrelationen > 0.7, was auf orthogonale (unabhängige) Signale hindeutet. Dies ist ein Vorteil 
für Interpretierbarkeit und Multikollinearitäts-Vermeidung.

---

## Vergleich RF vs. Linear

| Metrik | RF (Non-Linear) | Linear (ElasticNet) |
|--------|-----------------|---------------------|
| Total Biomarker | 21 | 4 |
| Proxy-Gruppen | 7 | 4 |
| Multi-Member Gruppen | 5 | 0 |
| Durchschn. Gruppengröße | 3.0 | 1.0 |
| Max. Intragruppen-Korr. | 0.91 | - |
| Hochkorrelierte Paare (r>0.8) | 10 | 0 |

---

## Key Findings

### 1. **RF identifiziert redundante Marker**
Der Random Forest Pfad findet mehr Biomarker, aber viele sind stark korreliert (redundant). 
Dies deutet auf biologische Pathway-Gruppen hin.

### 2. **Linear selektiert orthogonale Marker**
ElasticNet mit L1-Regularization erzwingt Sparsity und wählt automatisch unkorrelierte Features. 
Die 4 identifizierten Marker sind statistisch unabhängig.

### 3. **Methylierung ist stark strukturiert**
Die größte Proxy-Gruppe (n=8) besteht ausschließlich aus Methylierungsmarkern. Dies zeigt:
- Ko-Regulation von Methylierungsmustern
- Mögliche gemeinsame biologische Pathways
- Potenzial für dimensionale Reduktion (1 Representative statt 8)

### 4. **Praktische Implikationen**

**Für RF-Modelle:**
- Proxy-Gruppen können zur Feature-Reduktion verwendet werden
- Repräsentative reduzieren Modellkomplexität ohne Informationsverlust
- 21 → 7 Features (-67% Reduktion möglich)

**Für Linear-Modelle:**
- Keine Redundanz → alle 4 Features behalten
- Optimale Effizienz bereits erreicht

---

## Biologische Interpretation

### RF Methylierungs-Cluster

Die 8 hochkorrelierten Methylierungsmarker könnten repräsentieren:

1. **TMEM51** (Transmembrane protein): Membrantransport
2. **KLF4** (Krüppel-like factor 4): Stammzell-Transkriptionsfaktor
3. **TEAD4**: Hippo-Pathway, Zellproliferation
4. **DSTN** (Destrin): Aktin-Depolymerisation
5. **ZNF697** (Zinc finger): Transkriptionsregulation
6. **PTPN14**: Protein-Tyrosin-Phosphatase

**Hypothese**: Diese Gene könnten gemeinsam in einem epigenetischen Programm reguliert werden, 
das Zellproliferation, Differenzierung oder Therapieresistenz steuert.

### Linear Top-Marker: BCL2L1

**BCL2L1** (Bcl-xL) ist ein Anti-Apoptose-Faktor und bekannter Therapieresistenz-Marker.
- Support: 94 Medikamente (36.7%)
- Biologische Relevanz: Direkt mit Zelltod-Regulation assoziiert
- Therapeutisches Target: BCL-2 Inhibitoren (z.B. Navitoclax, ABT-737)

---

## Visualisierungen

Die generierten Plots befinden sich in:

### RF-Pfad:
- `results/proxy_analysis/rf/correlation_heatmap.png` - Korrelationsmatrix
- `results/proxy_analysis/rf/dendrogram.png` - Hierarchisches Clustering
- `results/proxy_analysis/rf/correlation_matrix.csv` - Vollständige Korrelationsmatrix

### Linear-Pfad:
- `results/proxy_analysis/linear/correlation_heatmap.png` - Korrelationsmatrix
- `results/proxy_analysis/linear/dendrogram.png` - Hierarchisches Clustering
- `results/proxy_analysis/linear/correlation_matrix.csv` - Vollständige Korrelationsmatrix

---

## Empfehlungen

### 1. Feature-Selektion für Downstream-Analysen
- **RF**: Verwende Representatives aus Proxy-Gruppen (7 statt 21 Features)
- **Linear**: Behalte alle 4 unabhängigen Marker

### 2. Biologische Validierung
Fokussiere auf:
- RF: METH__TMEM51-Cluster (größte Gruppe, r̄=0.80)
- Linear: TPM__BCL2L1 (höchster Support, biologisch relevant)

### 3. Modellverbesserung
- Teste dimensionale Reduktion mit Proxy-Repräsentativen
- Untersuche Pathway-Anreicherung für Methylierungs-Cluster
- Validiere BCL2L1 als prädiktiven Biomarker über alle Medikamente

---

**Generiert**: $(date)
**Parameter**: min_support=15, corr_threshold=0.7, max_drugs=100
