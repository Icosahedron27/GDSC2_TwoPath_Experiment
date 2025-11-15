"""Preprocessing pipeline - harmonize raw data to interim Parquet blocks."""
import sys
from pathlib import Path
from datetime import datetime
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.io_min import load_tpm, load_cnv, load_mut, load_methyl, load_ic50
from src.bachelorarbeit.prep.artifacts import save_block, save_table, save_manifest, save_sidecars


def main():
    cfg_dir = project_root / "configs" / "prep"
    with open(cfg_dir / "paths.yaml") as f:
        paths_cfg = yaml.safe_load(f)
    with open(cfg_dir / "global.yaml") as f:
        global_cfg = yaml.safe_load(f)
        
    raw_root = project_root / paths_cfg["root"]
    tpm_file = raw_root / paths_cfg["cmp"]["rnaseq_tpm"]["file"]
    cnv_file = raw_root / paths_cfg["cmp"]["cnv_summary"]["file"]
    mut_file = raw_root / paths_cfg["cmp"]["mutations_summary"]["file"]
    ic50_file = raw_root / paths_cfg["gdsc"]["dose_response"]["file"]
    gse_file = raw_root / paths_cfg["geo_gse68379"]["processed_matrix"]["file"]
    gpl_file = raw_root / paths_cfg["manifests"]["hm450k_manifest"]["file"]
    
    dtype = global_cfg.get('dtype', 'float32')
    outdir = project_root / "data" / "interim" / "transformed"
    outdir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "run_tag": global_cfg.get('run_tag', 'unknown'),
        "timestamp": datetime.now().isoformat(),
        "dtype": dtype,
        "blocks": {}
    }

    blocks_cfg = project_root / "configs" / "prep" / "blocks.yaml"
    
    # TPM
    print("Loading TPM...")
    tpm = load_tpm(tpm_file).astype(dtype)
    save_block(tpm, outdir / "tpm.parquet")
    save_sidecars(tpm, outdir / "tpm_sidecar")
    manifest["blocks"]["tpm"] = {"shape": list(tpm.shape), "dtype": str(tpm.dtypes.iloc[0])}
    
    # CNV
    print("Loading CNV...")
    cnv = load_cnv(cnv_file).astype(dtype)
    save_block(cnv, outdir / "cnv.parquet")
    save_sidecars(cnv, outdir / "cnv_sidecar")
    manifest["blocks"]["cnv"] = {"shape": list(cnv.shape), "dtype": str(cnv.dtypes.iloc[0])}
    
    # Mutations
    print("Loading Mutations...")
    mut = load_mut(mut_file, blocks_cfg).astype(dtype)
    save_block(mut, outdir / "mut.parquet")
    save_sidecars(mut, outdir / "mut_sidecar")
    manifest["blocks"]["mut"] = {"shape": list(mut.shape), "dtype": str(mut.dtypes.iloc[0])}
    
    # Methylation
    print("Loading Methylation...")
    methyl = load_methyl(gse_file, gpl_file, blocks_cfg).astype(dtype)
    save_block(methyl, outdir / "methyl.parquet")
    save_sidecars(methyl, outdir / "methyl_sidecar")
    manifest["blocks"]["methyl"] = {"shape": list(methyl.shape), "dtype": str(methyl.dtypes.iloc[0])}
    
    # IC50
    print("Loading IC50...")
    ic50 = load_ic50(ic50_file).astype(dtype)
    save_block(ic50, outdir / "ic50.parquet")
    manifest["blocks"]["ic50"] = {"shape": list(ic50.shape), "dtype": str(ic50.dtypes.iloc[0])}
    
    # Write manifest
    save_manifest(manifest, outdir / "manifest.json")
    print(f"\n✓ Saved {len(manifest['blocks'])} blocks to {outdir}")
    print(f"  Run tag: {manifest['run_tag']}")
    for block_name, info in manifest["blocks"].items():
        print(f"  {block_name}: {info['shape']} ({info['dtype']})")


if __name__ == "__main__":
    main()
