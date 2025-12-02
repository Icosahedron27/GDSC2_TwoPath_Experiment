"""Combined preprocessing and design-matrix pipeline for GDSC2 data."""
from pathlib import Path
import time
import yaml

# Load configs
configfile: "configs/prep/paths.yaml"
with open("configs/prep/global.yaml") as f:
    GLOBAL = yaml.safe_load(f)

RUN_ID = GLOBAL.get("run_tag", "default")

# Paths
RAW_ROOT = Path(config["root"])
INTERIM_DIR = Path("data/interim/transformed")
IC50_INTERIM = INTERIM_DIR / "ic50.parquet"
DRUG_LIST_FILE = INTERIM_DIR / "drugs_list.txt"

TPM_FILE = RAW_ROOT / config["cmp"]["rnaseq_tpm"]["file"]
CNV_FILE = RAW_ROOT / config["cmp"]["cnv_summary"]["file"]
MUT_FILE = RAW_ROOT / config["cmp"]["mutations_summary"]["file"]
IC50_FILE = RAW_ROOT / config["gdsc"]["dose_response"]["file"]
GSE_FILE = RAW_ROOT / config["geo_gse68379"]["processed_matrix"]["file"]
GPL_FILE = RAW_ROOT / config["manifests"]["hm450k_manifest"]["file"]

# Interim outputs
BLOCKS = ["tpm", "cnv", "mut", "methyl", "ic50"]
OUTPUTS = [str(INTERIM_DIR / f"{block}.parquet") for block in BLOCKS]
SIDECARS = [
    str(INTERIM_DIR / f"{block}_sidecar/cells.txt.gz")
    for block in ["tpm", "cnv", "mut", "methyl"]
]
MANIFEST = str(INTERIM_DIR / "manifest.json")
DESIGN_DONE = "results/.design_matrices_complete"

PIPELINE_TIMER = {"start": None}


def onstart():
    """Record pipeline start time and emit banner."""
    PIPELINE_TIMER["start"] = time.time()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(PIPELINE_TIMER["start"]))
    print(f"[PIPELINE] Started at {ts}")


def onsuccess():
    """Report pipeline completion and total wall-clock runtime."""
    end = time.time()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    start = PIPELINE_TIMER.get("start")
    if start is not None:
        duration = end - start
        print(f"[PIPELINE] Finished at {ts} (runtime {duration:.1f} s)")
    else:
        print(f"[PIPELINE] Finished at {ts}")


rule all:
    input:
        OUTPUTS,
        SIDECARS,
        MANIFEST,
        DESIGN_DONE


rule make_interim:
    # Build interim parquet blocks and sidecars from raw inputs.
    input:
        tpm=TPM_FILE,
        cnv=CNV_FILE,
        mut=MUT_FILE,
        ic50=IC50_FILE,
        gse=GSE_FILE,
        gpl=GPL_FILE,
        blocks_cfg="configs/prep/blocks.yaml",
        global_cfg="configs/prep/global.yaml"
    output:
        blocks=OUTPUTS,
        sidecars=SIDECARS,
        manifest=MANIFEST
    shell:
        "python3 scripts/prep_make_interim.py"


checkpoint collect_drugs:
    # Write the list of available drugs after IC50 aggregation.
    input:
        ic50=str(IC50_INTERIM)
    output:
        list_file=str(DRUG_LIST_FILE)
    run:
        import pandas as pd

        df = pd.read_parquet(input.ic50)
        drugs = sorted(col for col in df.columns if col != "cell_id")
        list_path = Path(output.list_file)
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(drugs))


def _select_drugs(drugs):
    """Apply optional subset filtering via config or global settings."""
    subset = config.get("design_drugs_subset") or GLOBAL.get("design_drugs_subset")
    if subset:
        if isinstance(subset, str):
            subset = [item.strip() for item in subset.split(",") if item.strip()]
        subset = set(subset)
        return [drug for drug in drugs if drug in subset]
    return drugs


def design_outputs(wildcards):
    """Expand all design-matrix artefacts for the current drug roster."""
    ckpt = checkpoints.collect_drugs.get()
    drugs = [
        line.strip()
        for line in Path(ckpt.output.list_file).read_text().splitlines()
        if line.strip()
    ]
    drugs = _select_drugs(drugs)
    if not drugs:
        return []
    templates = [
        "data/processed/{drug}/{run_id}/design_matrix/X.parquet",
        "data/processed/{drug}/{run_id}/design_matrix/y.parquet",
        "data/processed/{drug}/{run_id}/design_matrix/feature_meta.parquet",
        "data/processed/{drug}/{run_id}/design_matrix/manifest.json",
        "data/processed/{drug}/{run_id}/design_matrix/sidecars/cells.txt.gz",
        "data/processed/{drug}/{run_id}/design_matrix/sidecars/genes.txt.gz",
        "data/processed/{drug}/{run_id}/design_matrix/X_zScoreNormalized.parquet",
        "data/processed/{drug}/{run_id}/design_matrix/X_linear_sis_reduced.parquet",
        "data/processed/{drug}/{run_id}/design_matrix/X_general_sis_reduced.parquet",
        "results/{drug}/{run_id}/cpss_linear_scores.csv",
        "results/{drug}/{run_id}/cpss_rf_scores.csv",
    ]
    global_outputs = [
        "results/frequent_itemsets_rf.csv",
        "results/frequent_itemsets_linear.csv",
    ]
    return expand(templates, drug=drugs, run_id=RUN_ID) + global_outputs


rule design_matrices_complete:
    """Materialize a marker file once all design matrices are built."""
    input:
        design_outputs
    output:
        DESIGN_DONE
    run:
        marker = Path(output[0])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()


rule build_design_matrix:
    # Assemble per-drug design matrices from interim blocks.
    input:
        tpm=INTERIM_DIR / "tpm.parquet",
        cnv=INTERIM_DIR / "cnv.parquet",
        mut=INTERIM_DIR / "mut.parquet",
        methyl=INTERIM_DIR / "methyl.parquet",
        ic50=IC50_INTERIM,
        cfg="configs/prep/global.yaml"
    output:
        X="data/processed/{drug}/{run_id}/design_matrix/X.parquet",
        y="data/processed/{drug}/{run_id}/design_matrix/y.parquet",
        meta="data/processed/{drug}/{run_id}/design_matrix/feature_meta.parquet",
        manifest="data/processed/{drug}/{run_id}/design_matrix/manifest.json",
        cells="data/processed/{drug}/{run_id}/design_matrix/sidecars/cells.txt.gz",
        genes="data/processed/{drug}/{run_id}/design_matrix/sidecars/genes.txt.gz"
    shell:
        (
            "rm -rf 'data/processed/{wildcards.drug}/{wildcards.run_id}/design_matrix' "
            "&& python3 scripts/build_design_matrix.py --drug '{wildcards.drug}' --run-id '{wildcards.run_id}'"
        )

rule z_score_normalize:
    input:
        X="data/processed/{drug}/{run_id}/design_matrix/X.parquet",
        meta="data/processed/{drug}/{run_id}/design_matrix/feature_meta.parquet"
    output:
        X_norm="data/processed/{drug}/{run_id}/design_matrix/X_zScoreNormalized.parquet"
    shell:
        "python3 scripts/zscore_normalizer.py --drug '{wildcards.drug}' --run-id '{wildcards.run_id}'"

rule sis_linear:
    input:
        X_norm="data/processed/{drug}/{run_id}/design_matrix/X_zScoreNormalized.parquet",
        y="data/processed/{drug}/{run_id}/design_matrix/y.parquet"
    output:
        X_reduced="data/processed/{drug}/{run_id}/design_matrix/X_linear_sis_reduced.parquet"
    shell:
        "python3 scripts/sis.py --drug '{wildcards.drug}' --run-id '{wildcards.run_id}' linear"

rule sis_general:
    input:
        X_norm="data/processed/{drug}/{run_id}/design_matrix/X_zScoreNormalized.parquet",
        y="data/processed/{drug}/{run_id}/design_matrix/y.parquet"
    output:
        X_reduced="data/processed/{drug}/{run_id}/design_matrix/X_general_sis_reduced.parquet"
    shell:
        "python3 scripts/sis.py --drug '{wildcards.drug}' --run-id '{wildcards.run_id}' general"

rule cpss_linear:
    input:
        X="data/processed/{drug}/{run_id}/design_matrix/X_linear_sis_reduced.parquet",
        y="data/processed/{drug}/{run_id}/design_matrix/y.parquet"
    output:
        scores="results/{drug}/{run_id}/cpss_linear_scores.csv",
        significant="results/{drug}/{run_id}/cpss_linear_significant.csv",
        above_threshold="results/{drug}/{run_id}/cpss_linear_above_threshold.csv",
        above_wc="results/{drug}/{run_id}/cpss_linear_above_worst_case.csv",
        bounds="results/{drug}/{run_id}/cpss_linear_bounds.json"
    params:
        B=50
    shell:
        "python3 scripts/run_cpss.py --drug '{wildcards.drug}' --run-id '{wildcards.run_id}' --method linear --B {params.B}"

rule frequent_itemsets:
    input:
        lambda wildcards: expand("results/{drug}/v1-union-na20/cpss_rf_above_threshold.csv",
                                drug=[d.name for d in Path("results").iterdir() 
                                      if d.is_dir() and (d / "v1-union-na20").exists()])
    output:
        rf="results/frequent_itemsets_rf.csv",
        linear="results/frequent_itemsets_linear.csv"
    params:
        min_support=2,
        max_size=3,
        top_k=50
    shell:
        "python3 scripts/frequent_itemsets.py --method both --min-support {params.min_support} --max-size {params.max_size} --top-k {params.top_k}"

rule cpss_rf:
    input:
        X="data/processed/{drug}/{run_id}/design_matrix/X_general_sis_reduced.parquet",
        y="data/processed/{drug}/{run_id}/design_matrix/y.parquet"
    output:
        scores="results/{drug}/{run_id}/cpss_rf_scores.csv",
        significant="results/{drug}/{run_id}/cpss_rf_significant.csv",
        above_threshold="results/{drug}/{run_id}/cpss_rf_above_threshold.csv",
        above_wc="results/{drug}/{run_id}/cpss_rf_above_worst_case.csv",
        bounds="results/{drug}/{run_id}/cpss_rf_bounds.json"
    params:
        B=50
    shell:
        "python3 scripts/run_cpss.py --drug '{wildcards.drug}' --run-id '{wildcards.run_id}' --method rf --B {params.B}"