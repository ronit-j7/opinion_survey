"""Step 01 logic: load the survey CSV, encode answers, apply the filter cascade."""
from typing import Literal

import numpy as np
import pandas as pd

from rn import config

Tier = Literal["all", "ge40", "complete"]


def split_column(col: str) -> tuple[str, str]:
    """'T01. Artificial Intelligence will ...' -> ('T01', '...')."""
    code, text = col.split(".", 1)
    return code.strip(), text.strip()


def load_encoded(path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (responses, items, info).

    responses: response_id x items, float 1-5, NaN = blank or "No Comments".
    items:     code, theme, text, per-item response counts.
    info:      cascade counts and the response_id -> row_position map.
    """
    raw = pd.read_csv(path)
    row_position = {str(rid): int(pos) for pos, rid in enumerate(raw[raw.columns[0]])}
    raw = raw.set_index(raw.columns[0])
    raw.index.name = "response_id"

    codes, texts = zip(*(split_column(c) for c in raw.columns))
    raw.columns = list(codes)

    answers = raw.apply(lambda s: s.astype("string").str.strip())
    allowed = set(config.LIKERT) | {config.NO_COMMENT}
    seen = set(pd.unique(answers.stack().dropna()))
    unknown = seen - allowed
    if unknown:
        raise ValueError(f"Unexpected answer values: {sorted(unknown)}")

    responses = answers.apply(lambda s: s.map(config.LIKERT)).astype(float)
    responses = responses.loc[~responses.isna().all(axis=1)]

    items = pd.DataFrame({
        "code": codes,
        "theme": [c[0] for c in codes],
        "text": texts,
        "n_responses": responses.notna().sum().to_numpy(),
        "n_no_comment": (answers == config.NO_COMMENT).sum().to_numpy(),
    })
    info = {
        "n_rows_in_file": len(raw),
        "n_nonblank": len(responses),
        "row_position": row_position,
    }
    return responses, items, info


def tier_mask(responses: pd.DataFrame, tier: Tier) -> pd.Series:
    """Boolean mask over respondents for a filter tier."""
    if tier == "all":
        return responses.notna().any(axis=1)
    if tier == "ge40":
        return responses.notna().sum(axis=1) >= config.MIN_ANSWERED
    if tier == "complete":
        return responses.notna().all(axis=1)
    raise ValueError(f"unknown tier {tier}")


def apply_tier(responses: pd.DataFrame, tier: Tier) -> pd.DataFrame:
    return responses.loc[tier_mask(responses, tier)]


def domain_columns(responses: pd.DataFrame) -> dict[str, list[str]]:
    return {d: [c for c in responses.columns if c[0] == d] for d in config.DOMAINS}


def subset_stats(X: pd.DataFrame) -> dict:
    """Diagnostic summary of one respondent subset."""
    st = X.stack()
    agree = (st >= 4).mean()
    dis = (X <= 2).mean()
    doms = domain_columns(X)
    dmeans = {d: float(X[cols].mean(axis=1).mean()) for d, cols in doms.items()}
    dm = pd.DataFrame({d: X[cols].mean(axis=1) for d, cols in doms.items()})
    return {
        "n": len(X),
        "agree_side_share": float(agree),
        "items_under_5pct_disagreement": int((dis < 0.05).sum()),
        "E03_disagree_share": float((X["E03"] <= 2).mean()),
        "E02_disagree_share": float((X["E02"] <= 2).mean()),
        "domain_means": dmeans,
        "corr_S_V": float(dm["S"].corr(dm["V"])),
        "corr_T_E": float(dm["T"].corr(dm["E"])),
    }
