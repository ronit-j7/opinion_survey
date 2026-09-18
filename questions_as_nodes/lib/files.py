"""Reading and writing the intermediate files each step hands to the next."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_matrix(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, float_format="%.10g")


def read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def read_responses(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def read_items(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(_to_builtin(obj), indent=2))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())
