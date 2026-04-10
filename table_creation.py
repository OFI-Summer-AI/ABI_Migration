"""
Generate Databricks CREATE TABLE DDL for KPI result tables (schema only, no data).

For each KPI in data/raw/analysis_kpis.json:
- Infer process domain (sales order, delivery, shipment, invoice) and include
  primary-key grain columns for the standard SAP header/item tables.
- Add every \"TABLE\".\"COLUMN\" reference found in pql_formula / pql_formula_expanded.
- Add one column per nested KPI(\"...\") flag from the (non-expanded) PQL formula.
- Add the KPI's own output column (same name as kpi_id).

Output: data/translation/kpi_tables_create_only.sql
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dependency.kpi_dependency_graph import KPI_CALL_RE  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRANSLATION_DIR = PROJECT_ROOT / "data" / "translation"
ANALYSIS_KPIS_PATH = RAW_DIR / "analysis_kpis.json"

DATABRICKS_CATALOG = "main"
DATABRICKS_SCHEMA = "touchless_kpis"

# \"Table\".\"Column\" as extracted from Celonis / PQL text
TABLE_COL_RE = re.compile(r'"([^"]+)"\s*\.\s*"([^"]+)"')

Domain = str

# Per domain: SAP tables and their grain / key columns (always present in DDL when domain applies)
DOMAIN_TABLE_KEYS: Dict[Domain, List[Tuple[str, str]]] = {
    "sales_order": [
        ("VBAK", "VBELN"),
        ("VBAP", "VBELN"),
        ("VBAP", "POSNR"),
    ],
    "delivery": [
        ("LIKP", "VBELN"),
        ("LIPS", "VBELN"),
        ("LIPS", "POSNR"),
    ],
    "shipment": [
        ("VTTK", "TKNUM"),
        ("VTTP", "TKNUM"),
        ("VTTP", "TPNUM"),
    ],
    "invoice": [
        ("VBRK", "VBELN"),
        ("VBRP", "VBELN"),
        ("VBRP", "POSNR"),
    ],
}


def load_analysis_kpis(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing KPI source: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def dedupe_kpis(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("attribute_type", "")).lower() != "kpi":
            continue
        kid = str(row.get("kpi_id", "") or "").strip()
        if not kid or kid in seen:
            continue
        seen.add(kid)
        out.append(row)
    return out


def sql_ident(s: str) -> str:
    """Databricks-style escaped identifier."""
    return "`" + s.replace("`", "``") + "`"


def physical_col_name(table: str, column: str) -> str:
    return f"{table.lower()}_{column.lower()}"


def extract_table_columns(pql: str) -> List[Tuple[str, str]]:
    """Return (TABLE, COLUMN) with COLUMN uppercased for stable deduplication."""
    pairs: List[Tuple[str, str]] = []
    for t, c in TABLE_COL_RE.findall(pql or ""):
        pairs.append((t.strip().upper(), c.strip().upper()))
    return pairs


def tables_from_pairs(pairs: List[Tuple[str, str]]) -> Set[str]:
    return {t for t, _ in pairs}


def ordered_nested_kpi_ids(pql: str) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for m in KPI_CALL_RE.finditer(pql or ""):
        ref = (m.group(1) or m.group(2) or "").strip()
        if ref and ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def classify_domain(kpi_id: str, kpi_name: str, tables: Set[str]) -> Domain:
    """Pick one domain: invoice, shipment, delivery, or sales_order."""
    kid = kpi_id.lower()
    name = (kpi_name or "").lower()
    t = {x.upper() for x in tables}

    if (
        "vbrk" in t
        or "vbrp" in t
        or "invoice" in kid
        or "invoice" in name
        or "cash_application" in kid
        or "cash application" in name
    ):
        return "invoice"

    if "vttk" in t or "vttp" in t or "shipment" in kid:
        return "shipment"

    if "likp" in t or "lips" in t or "delivery" in kid:
        return "delivery"

    if "vbak" in t or "vbap" in t or "order" in kid:
        return "sales_order"

    return "sales_order"


def merge_required_columns(
    domain: Domain,
    pairs_from_pql: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Union of (1) domain primary keys and (2) every table.column referenced in PQL text.
    Order: domain keys first (definition order), then remaining refs sorted by table, column.
    Keys are (TABLE, COLUMN) with COLUMN uppercased so VBRP.VBELN and VBRP.vbeln map once.
    """
    seen: Set[Tuple[str, str]] = set()
    ordered: List[Tuple[str, str]] = []

    for tbl, col in DOMAIN_TABLE_KEYS.get(domain, []):
        key = (tbl.upper(), col.upper())
        if key not in seen:
            seen.add(key)
            ordered.append(key)

    rest: List[Tuple[str, str]] = []
    for tbl, col in pairs_from_pql:
        key = (tbl.upper(), col.upper())
        if key not in seen:
            seen.add(key)
            rest.append(key)
    rest.sort(key=lambda x: (x[0], x[1]))
    ordered.extend(rest)
    return ordered


def build_create_table_statement(
    *,
    kpi_id: str,
    display_name: str,
    pql_formula: str,
    expanded_pql: str,
    catalog: str,
    schema: str,
) -> str:
    flags = ordered_nested_kpi_ids(pql_formula)
    pql_blob = f"{pql_formula or ''}\n{expanded_pql or ''}"
    pairs = extract_table_columns(pql_blob)
    domain = classify_domain(kpi_id, display_name, tables_from_pairs(pairs))
    base_cols = merge_required_columns(domain, pairs)

    phys_names: List[str] = []
    col_lines: List[str] = []
    for tbl, col in base_cols:
        phys = physical_col_name(tbl, col)
        phys_names.append(phys)
        col_lines.append(
            f"  {sql_ident(phys)} STRING COMMENT '{tbl}.{col}'"
        )

    for fid in flags:
        if fid in phys_names:
            continue
        col_lines.append(
            f"  {sql_ident(fid)} STRING COMMENT 'Nested KPI flag; value from PQL/SQL for {fid}'"
        )

    kpi_col = kpi_id
    if kpi_col not in flags and kpi_col not in phys_names:
        col_lines.append(
            f"  {sql_ident(kpi_col)} STRING COMMENT 'KPI output: {display_name}'"
        )
    elif kpi_col in flags or kpi_col in phys_names:
        out_col = f"{kpi_col}_value"
        col_lines.append(
            f"  {sql_ident(out_col)} STRING COMMENT 'KPI output: {display_name} (renamed; {kpi_col} already used)'"
        )

    fq = f"{sql_ident(catalog)}.{sql_ident(schema)}.{sql_ident(kpi_id)}"
    meta = [
        "/* --------------------------------------------------------------------",
        f" * Table: {kpi_id}",
        f" * Display name: {display_name}",
        f" * Domain: {domain}",
        f" * Nested KPI flags: {', '.join(flags) if flags else '(none)'}",
        " * -------------------------------------------------------------------- */",
    ]
    body = [
        f"CREATE TABLE IF NOT EXISTS {fq} (",
        ",\n".join(col_lines),
        f") COMMENT {json.dumps(display_name)};",
        "",
    ]
    return "\n".join(meta + body)


def main() -> None:
    rows = load_analysis_kpis(ANALYSIS_KPIS_PATH)
    kpis = dedupe_kpis(rows)
    if not kpis:
        raise SystemExit(f"No KPI rows after dedupe in {ANALYSIS_KPIS_PATH}")

    TRANSLATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRANSLATION_DIR / "kpi_tables_create_only.sql"

    preamble = "\n".join(
        [
            "/* ============================================================================",
            " * kpi_tables_create_only.sql — generated by table_creation.py",
            " *",
            " * Column layout per KPI:",
            " * - Primary grain + keys for domain tables (VBAK/VBAP, LIKP/LIPS, VTTK/VTTP, VBRK/VBRP)",
            " * - All TABLE.COLUMN references found in pql_formula / pql_formula_expanded",
            " * - One STRING column per KPI(\\\"...\\\") nested flag in the base PQL",
            " * - Final STRING column for the KPI result",
            f" * Target catalog.schema: {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}",
            " * ============================================================================ */",
            "",
        ]
    )

    blocks: List[str] = []
    for row in kpis:
        kpi_id = str(row.get("kpi_id", "") or "").strip()
        name = str(row.get("name", "") or "")
        pql = str(row.get("pql_formula", "") or "")
        expanded = str(row.get("pql_formula_expanded", "") or "")
        blocks.append(
            build_create_table_statement(
                kpi_id=kpi_id,
                display_name=name,
                pql_formula=pql,
                expanded_pql=expanded,
                catalog=DATABRICKS_CATALOG,
                schema=DATABRICKS_SCHEMA,
            )
        )

    out_path.write_text(preamble + "\n".join(blocks), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
