
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_configured_path = os.environ.get("IBTRACS_CSV_PATH")
DATA_PATH = Path(_configured_path).expanduser() if _configured_path else PROJECT_ROOT / "Data" / "ibtracs.csv"
if not DATA_PATH.is_absolute():
    DATA_PATH = PROJECT_ROOT / DATA_PATH


def load_cyclone_data(path: Path = DATA_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"IBTrACS CSV not found at {dataset_path}. "
            "Place ibtracs.csv in the project's Data folder or set IBTRACS_CSV_PATH."
        )
    if dataset_path.stat().st_size == 0:
        raise ValueError(f"IBTrACS CSV is empty: {dataset_path}")

    return pd.read_csv(dataset_path, skiprows=[1], low_memory=False)
