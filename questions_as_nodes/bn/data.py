"""Step 01: load the survey CSV and encode answers as 1-5."""
from pathlib import Path

import numpy as np
import pandas as pd

from bn import config


def split_column(col: str) -> tuple[str, str]:
    """'T01. Artificial Intelligence will ...' -> ('T01', 'Artificial Intelligence will ...')."""
    code, text = col.split(".", 1)
    return code.strip(), text.strip()


def load_responses(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (responses, items, info).

    responses: respondents x items, float 1-5, NaN = missing (blank or "No Comments").
               Respondents with no observed answers are dropped.
    items:     one row per item with code, theme, statement text and response counts.
    """
    raw = pd.read_csv(path)
    raw = raw.set_index(raw.columns[0])
    raw.index.name = "response_id"

    codes, texts = zip(*(split_column(c) for c in raw.columns))
    raw.columns = list(codes)

    answers = raw.apply(lambda s: s.astype("string").str.strip())
    allowed = set(config.LIKERT) | {config.NO_COMMENT}
    seen = set(pd.unique(answers.stack().dropna()))
    unknown = seen - allowed
    if unknown:
        raise ValueError(f"Unexpected answer values in {path.name}: {sorted(unknown)}")

    responses = answers.apply(lambda s: s.map(config.LIKERT)).astype(float)

    empty = responses.isna().all(axis=1)
    responses = responses.loc[~empty]
    answers = answers.loc[~empty]

    items = pd.DataFrame({
        "code": codes,
        "theme": [c[0] for c in codes],
        "theme_name": [config.THEMES[c[0]] for c in codes],
        "text": texts,
        "n_responses": responses.notna().sum().to_numpy(),
        "n_no_comment": (answers == config.NO_COMMENT).sum().to_numpy(),
    })

    info = {
        "n_rows_in_file": len(raw),
        "n_empty_rows_dropped": int(empty.sum()),
        "n_respondents": len(responses),
        "n_items": responses.shape[1],
        "n_complete_cases": int(responses.notna().all(axis=1).sum()),
        "n_no_comment_cells": int(items["n_no_comment"].sum()),
        "n_missing_cells": int(np.isnan(responses.to_numpy()).sum()),
    }
    return responses, items, info
