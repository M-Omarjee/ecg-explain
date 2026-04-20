"""SCP code to diagnostic superclass mapping for PTB-XL.

PTB-XL annotates each ECG with one or more SCP-ECG statements (e.g. 'IMI',
'ASMI', 'NORM'). For the standard 5-class benchmark, these are aggregated
into 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_TO_IDX = {c: i for i, c in enumerate(SUPERCLASSES)}


def load_scp_mapping(scp_statements_path: Path) -> dict[str, str]:
    """Load mapping from SCP code -> diagnostic superclass.

    Only includes statements where diagnostic == 1 (i.e. proper diagnostic
    statements, not form/rhythm-only labels).
    """
    df = pd.read_csv(scp_statements_path, index_col=0)
    df = df[df["diagnostic"] == 1]
    return df["diagnostic_class"].dropna().to_dict()


def parse_scp_codes(scp_codes_str: str) -> dict[str, float]:
    """Parse the scp_codes column from ptbxl_database.csv.

    The column stores stringified Python dicts, e.g. "{'NORM': 100.0, 'SR': 0.0}".
    """
    return ast.literal_eval(scp_codes_str)


def scp_to_superclass_labels(
    scp_codes_str: str,
    scp_mapping: dict[str, str],
) -> np.ndarray:
    """Convert SCP codes for a single record to a multi-hot superclass vector."""
    codes = parse_scp_codes(scp_codes_str)
    labels = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for code in codes:
        superclass = scp_mapping.get(code)
        if superclass and superclass in SUPERCLASS_TO_IDX:
            labels[SUPERCLASS_TO_IDX[superclass]] = 1.0
    return labels
