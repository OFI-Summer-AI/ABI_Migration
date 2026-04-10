"""
Print a terminal-friendly view of KPI table schemas produced by table_creation.py.

Uses the same rules as table_creation (analysis_kpis.json, domain keys, PQL columns, flags).
Run: python tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import table_creation as tc  # noqa: E402


def domain_key_set(domain: str) -> Set[Tuple[str, str]]:
    return {(t.upper(), c.upper()) for t, c in tc.DOMAIN_TABLE_KEYS.get(domain, [])}


def kpi_column_rows(
    kpi_id: str,
    display_name: str,
    pql_formula: str,
    expanded_pql: str,
) -> Tuple[str, List[Dict[str, str]]]:
    flags = tc.ordered_nested_kpi_ids(pql_formula)
    pql_blob = f"{pql_formula or ''}\n{expanded_pql or ''}"
    pairs = tc.extract_table_columns(pql_blob)
    domain = tc.classify_domain(kpi_id, display_name, tc.tables_from_pairs(pairs))
    base_cols = tc.merge_required_columns(domain, pairs)
    dkeys = domain_key_set(domain)

    rows: List[Dict[str, str]] = []
    phys_seen: List[str] = []

    for tbl, col in base_cols:
        phys = tc.physical_col_name(tbl, col)
        phys_seen.append(phys)
        key = (tbl.upper(), col.upper())
        role = "grain" if key in dkeys else "from_pql"
        rows.append(
            {
                "column": phys,
                "type": "STRING",
                "role": role,
                "source": f"{tbl}.{col}",
            }
        )

    for fid in flags:
        if fid in phys_seen:
            continue
        rows.append(
            {
                "column": fid,
                "type": "STRING",
                "role": "nested_kpi",
                "source": f'KPI("{fid}")',
            }
        )

    kpi_col = kpi_id
    if kpi_col not in flags and kpi_col not in phys_seen:
        out_name = kpi_col
    else:
        out_name = f"{kpi_col}_value"
    rows.append(
        {
            "column": out_name,
            "type": "STRING",
            "role": "kpi_output",
            "source": display_name or kpi_id,
        }
    )

    return domain, rows


def _cell(s: str, width: int) -> str:
    s = s.replace("\n", " ")
    if len(s) <= width:
        return s.ljust(width)
    if width <= 3:
        return s[:width].ljust(width)
    return (s[: width - 3] + "...")[:width].ljust(width)


def print_box_table(title: str, subtitle: str, headers: List[str], data_rows: List[List[str]]) -> None:
    cols = len(headers)
    max_cell = 52
    widths = [len(h) for h in headers]
    for r in data_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], min(len(cell), max_cell))
    if data_rows:
        widths[0] = max(widths[0], len(str(len(data_rows))))

    def sep_line() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row_line(cells: List[str]) -> str:
        parts = [f" {_cell(cells[i], widths[i])} " for i in range(cols)]
        return "|" + "|".join(parts) + "|"

    tw = len(sep_line())  # total box width

    print(sep_line())
    print("| " + _cell(title, tw - 4) + " |")
    print(sep_line())
    for sub in _wrap_line(subtitle, tw - 4):
        print("| " + _cell(sub, tw - 4) + " |")
    print(sep_line())
    print(row_line(headers))
    print(sep_line())
    for r in data_rows:
        print(row_line(r))
    print(sep_line())
    print()


def _wrap_line(text: str, width: int) -> List[str]:
    if width < 8:
        return [text[: max(1, width - 3)] + "..."] if len(text) >= width else [text]
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def print_schema_tables() -> None:
    rows = tc.load_analysis_kpis(tc.ANALYSIS_KPIS_PATH)
    kpis = tc.dedupe_kpis(rows)
    if not kpis:
        print("No KPI rows found.", file=sys.stderr)
        sys.exit(1)

    fq = f"{tc.DATABRICKS_CATALOG}.{tc.DATABRICKS_SCHEMA}"
    banner = (
        "KPI table schemas (same layout as table_creation.py -> kpi_tables_create_only.sql)\n"
        f"Target: {fq}.*"
    )
    print()
    print("=" * min(88, max(len(banner), 40)))
    print(banner)
    print("=" * min(88, max(len(banner), 40)))
    print()

    headers = ["#", "Column", "Type", "Role", "Source / comment"]

    for idx, row in enumerate(kpis, start=1):
        kpi_id = str(row.get("kpi_id", "") or "").strip()
        name = str(row.get("name", "") or "")
        pql = str(row.get("pql_formula", "") or "")
        expanded = str(row.get("pql_formula_expanded", "") or "")
        domain, col_rows = kpi_column_rows(kpi_id, name, pql, expanded)

        title = f"{kpi_id}"
        subtitle = (
            f"Display: {name}  |  Domain: {domain}  |  "
            f"Full name: {tc.sql_ident(tc.DATABRICKS_CATALOG)}."
            f"{tc.sql_ident(tc.DATABRICKS_SCHEMA)}.{tc.sql_ident(kpi_id)}"
        )
        data: List[List[str]] = []
        for i, c in enumerate(col_rows, start=1):
            data.append(
                [
                    str(i),
                    c["column"],
                    c["type"],
                    c["role"],
                    c["source"],
                ]
            )
        print_box_table(title, subtitle, headers, data)

    print(f"Total KPI tables: {len(kpis)}")


def main() -> None:
    print_schema_tables()


if __name__ == "__main__":
    main()
