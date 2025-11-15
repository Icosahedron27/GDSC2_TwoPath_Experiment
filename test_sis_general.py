from pathlib import Path
from scripts.sis import sis_general

project_root = Path(__file__).resolve().parent
design_matrix_path = project_root / 'data/processed/Docetaxel/v1-union-na20/design_matrix'

print("Testing SIS General for Docetaxel...")
sis_general(design_matrix_path, alpha=4.0)
