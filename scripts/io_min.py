import pandas as pd
import yaml
from pathlib import Path


def load_tpm(file_path: Path) -> pd.DataFrame:
    """
    Load TPM data from a dictionary file.
    Output: DataFrame with TPM values for genes and cell types.

    data format:
    The input file is a CSV file with the following structure:
    - First row for model IDs starting at after third comma
    - Second row for model names starting at after third comma
    - Third row is skipped
    - Fourth row shows general structure: <gene_id>, <cell_line_1>, <cell_line_2>, (without TPM values)
    - From fifth row onwards: <gene_symbol>, <ensembl_id>, <gene_id>, <TPM_value_1>, <TPM_value_2>, ...
    """
    df_header = pd.read_csv(file_path, nrows=2, header=None, sep = ',')
    df_data = pd.read_csv(file_path, skiprows=3)

    # Use model_name (row 2) instead of model_id for consistency across data sources
    model_names = df_header.iloc[1, 3:].values

    gene_symbols = df_data.iloc[:, 0].values
    tpm_matrix_pre = df_data.iloc[:, 3:].values

    tpm_matrix = pd.DataFrame(
        tpm_matrix_pre.T,
        index=model_names,
        columns=gene_symbols
    )

    tpm_matrix = tpm_matrix.apply(pd.to_numeric, errors='coerce')
    if tpm_matrix.columns.has_duplicates:
        # Collapse duplicated gene symbols by averaging across the duplicates
        tpm_matrix = tpm_matrix.groupby(level=0, axis=1, sort=False).mean()

    tpm_matrix = tpm_matrix.round(2)

    return tpm_matrix

def load_cnv(file_path: Path) -> pd.DataFrame:
    """
    Load CNV (Copy Number Variation) data and convert.
    """
    df = pd.read_csv(file_path)
    
    df_cnv = df.pivot_table(
        index='model_name',
        columns='symbol',
        values='total_copy_number',
        aggfunc='mean'
    )
    df_cnv = df_cnv.round(2)

    return df_cnv

def load_mut(file_path: Path, config_path: Path = None) -> pd.DataFrame:
    """
    Load mutation data and convert to weighted mutation matrix.
    """
    with open(config_path, 'r') as f:
        blocks = yaml.safe_load(f)
    
    mutation_weights = blocks.get("mutation_weights")
    default_weight = mutation_weights.pop("else", 0)
    
    df = pd.read_csv(file_path, usecols=["model_name", "gene_symbol", "effect", "vaf"])

    df["weight"] = df["effect"].map(mutation_weights).fillna(default_weight)
    
    df["vaf"] = pd.to_numeric(df["vaf"], errors='coerce').fillna(1.0)
    df["mutation_score"] = df["weight"] * df["vaf"]
    
    df_mut = df.pivot_table(
        index='model_name',
        columns='gene_symbol',
        values='mutation_score',
        aggfunc='max',
        fill_value=0.0
    )
    
    df_mut = df_mut.round(2)
    
    return df_mut
    
def load_methyl(gse_path: Path, gpl_path: Path, config_path: Path = None) -> pd.DataFrame:
    """
    Load methylation data filtered by TSS1500/TSS200 regions.
    """
    with open(config_path, 'r') as f:
        blocks = yaml.safe_load(f)
    
    interested_groups = blocks.get("interestedRefgeneGroups")
    gpl = pd.read_csv(gpl_path, sep="\t", comment="#", low_memory=False)
    gpl_filtered = gpl[gpl["UCSC_RefGene_Group"].str.contains("|".join(interested_groups), na=False)].copy()
    gpl_filtered["gene_symbol"] = gpl_filtered["UCSC_RefGene_Name"].str.split(";")
    gpl_filtered = gpl_filtered.explode("gene_symbol")
    gpl_filtered = gpl_filtered[gpl_filtered["gene_symbol"].notna() & (gpl_filtered["gene_symbol"] != "")]
    probe_gene_map = gpl_filtered[["ID", "gene_symbol"]].rename(columns={"ID": "Probe_ID"})
    methyl = pd.read_csv(gse_path, sep="\t")
    beta_cols = [col for col in methyl.columns if col.endswith("_AVG.Beta")]
    methyl_beta = methyl[["Row.names"] + beta_cols].rename(columns={"Row.names": "Probe_ID"})
    methyl_beta.columns = ["Probe_ID"] + [col.replace("_AVG.Beta", "") for col in beta_cols]
    methyl_annotated = probe_gene_map.merge(methyl_beta, on="Probe_ID", how="inner")
    methyl_annotated = methyl_annotated.drop(columns=["Probe_ID"])
    methyl_gene = methyl_annotated.groupby("gene_symbol").mean()
    methyl_transposed = methyl_gene.T
    methyl_final = methyl_transposed.round(2)
    
    return methyl_final

def load_ic50(file_path: Path) -> pd.DataFrame:
    """
    Load IC50 drug response data.
    """
    df = pd.read_excel(file_path)
    
    df_ic50 = df.pivot_table(
        index='CELL_LINE_NAME',
        columns='DRUG_NAME',
        values='LN_IC50',
        aggfunc='mean'
    )
    
    df_ic50 = df_ic50.round(2)

    df_ic50.index = df_ic50.index.astype(str).str.strip()
    df_ic50.columns = df_ic50.columns.astype(str).str.strip()

    if df_ic50.columns.duplicated().any():
        df_ic50 = df_ic50.groupby(level=0, axis=1, sort=False).mean()
    
    return df_ic50
