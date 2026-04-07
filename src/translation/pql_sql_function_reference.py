"""
Load PQL → SQL function mapping from Excel (data/raw/PQL_to_SQL_Function_Reference.xlsx).

Used as LLM context for sql_translation.py. The workbook can use any sheet; the first
sheet is read. First two columns are treated as PQL construct / SQL equivalent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


def load_pql_sql_function_reference(
    path: Path,
    *,
    max_rows: int = 120,
    sheet_name: Any = 0,
) -> str:
    """
    Return a compact bullet list for system prompts, or empty string if file missing/unreadable.
    """
    if path is None or not path.exists():
        return ""

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=0, engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=0)
        except Exception:
            return ""

    if df is None or df.empty or len(df.columns) < 2:
        return ""

    col_a, col_b = df.columns[0], df.columns[1]
    lines: List[str] = []
    for _, row in df.iterrows():
        a, b = row.get(col_a), row.get(col_b)
        if pd.isna(a) or pd.isna(b):
            continue
        sa, sb = str(a).strip(), str(b).strip()
        if not sa or not sb:
            continue
        lines.append(f"- {sa} → {sb}")
        if len(lines) >= max_rows:
            break

    if not lines:
        return ""

    header = (
        "PQL-to-SQL function mapping (from project reference workbook; "
        "use these mappings when translating PQL to Spark/SQL):\n"
    )
    return header + "\n".join(lines)


def default_reference_path(project_root: Path) -> Path:
    return project_root / "data" / "raw" / "PQL_to_SQL_Function_Reference.xlsx"
