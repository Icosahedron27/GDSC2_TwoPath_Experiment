"""Artifact persistence module - save/load Parquet with sidecars."""
import gzip
import json
from pathlib import Path
import pandas as pd


def save_block(df: pd.DataFrame, path: Path) -> None:
    """Save wide matrix (cells × features) as Parquet with cell_id as column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.reset_index(names='cell_id')
    df_out.to_parquet(path, index=False, engine='pyarrow')


def load_block(path: Path, columns: list = None) -> pd.DataFrame:
    """Load matrix with optional column selection, restore cell_id as index."""
    if columns:
        columns = ['cell_id'] + columns
    df = pd.read_parquet(path, columns=columns, engine='pyarrow')
    return df.set_index('cell_id')


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save tidy table without index (e.g., IC50, y, metadata)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine='pyarrow')


def save_manifest(data: dict, path: Path) -> None:
    """Save manifest.json atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_sidecars(df: pd.DataFrame, outdir: Path) -> None:
    """Save cells.txt.gz and genes.txt.gz for quick inspection."""
    outdir.mkdir(parents=True, exist_ok=True)
    
    cells_path = outdir / "cells.txt.gz"
    with gzip.open(cells_path, 'wt') as f:
        f.write('\n'.join(df.index.astype(str)))

    genes_path = outdir / "genes.txt.gz"
    with gzip.open(genes_path, 'wt') as f:
        f.write('\n'.join(df.columns.astype(str)))
