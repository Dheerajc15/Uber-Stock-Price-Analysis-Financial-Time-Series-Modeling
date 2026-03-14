from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW       = PROJECT_ROOT / "data"    / "raw"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES  = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"


def ensure_dirs() -> None:
    for p in [DATA_RAW, OUTPUT_FIGURES, OUTPUT_TABLES, OUTPUT_METRICS]:
        p.mkdir(parents=True, exist_ok=True)