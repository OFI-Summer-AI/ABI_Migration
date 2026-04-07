"""
Shared helpers for SQL DDL generation (dummy.py and similar).

- Table catalog: Celonis technical table_name vs optional table_alias.
  Convention: when ``table_alias`` is set, it matches the identifier used in Databricks / PQL.
- JOIN lines: use ``foreign_keys`` / table_dependency_graph edges (systematic keys),
  not extracted_transformations.json (supplemental only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TableCatalog:
    """Maps Celonis technical names and PQL identifiers to Databricks table names."""

    # UPPER(technical_table_name) or UPPER(alias) -> databricks_table_name (alias or technical)
    upper_to_db: Dict[str, str] = field(default_factory=dict)

    def databricks_name(self, pql_or_tech_fragment: str) -> str:
        raw = (pql_or_tech_fragment or "").strip()
        if not raw:
            return raw
        u = raw.upper()
        return self.upper_to_db.get(u, raw)

    @staticmethod
    def from_data_model_metadata(metadata: Optional[Dict[str, Any]]) -> "TableCatalog":
        cat = TableCatalog()
        if not isinstance(metadata, dict):
            return cat
        for t in metadata.get("tables") or []:
            if not isinstance(t, dict):
                continue
            tech = str(t.get("table_name") or "").strip()
            alias = str(t.get("table_alias") or "").strip()
            db = alias if alias else tech
            if not tech and not alias:
                continue
            if not db:
                continue
            if tech:
                cat.upper_to_db[tech.upper()] = db
            if alias:
                cat.upper_to_db[alias.upper()] = db
        return cat


def _edge_db_names(
    edge: Dict[str, Any],
    catalog: TableCatalog,
) -> Tuple[str, str]:
    src_t = str(edge.get("from") or "")
    tgt_t = str(edge.get("to") or "")
    return catalog.databricks_name(src_t), catalog.databricks_name(tgt_t)


def build_join_clause(
    driver_db: str,
    needed_db_tables: Set[str],
    edges: List[Dict[str, Any]],
    catalog: TableCatalog,
) -> Tuple[str, List[str]]:
    """
    Build FROM ... LEFT JOIN lines using FK edges between tables that appear in the KPI.

    FK direction in metadata: source (from) -> target (to); column_mappings list
    source_column_name on ``from``, target_column_name on ``to``.

    Returns (sql_fragment, notes). sql_fragment starts with 'FROM ...' and includes joins.
    """
    needed = {driver_db} | {t for t in needed_db_tables if t}
    connected: Set[str] = {driver_db}
    remaining = set(needed) - connected
    join_lines: List[str] = [f"FROM `{driver_db}` AS `{driver_db}`"]
    notes: List[str] = []

    if not remaining:
        return "\n".join(join_lines), notes

    def join_cond(db_new: str, db_anchor: str, edge: Dict[str, Any], new_is_source: bool) -> str:
        parts: List[str] = []
        cols = edge.get("columns") or []
        for cm in cols:
            if not isinstance(cm, dict):
                continue
            sc = str(cm.get("source_column_name") or "").strip()
            tc = str(cm.get("target_column_name") or "").strip()
            if not sc or not tc:
                continue
            if new_is_source:
                parts.append(f"`{db_new}`.`{sc}` = `{db_anchor}`.`{tc}`")
            else:
                parts.append(f"`{db_anchor}`.`{sc}` = `{db_new}`.`{tc}`")
        return " AND ".join(parts) if parts else ""

    safety = 0
    while remaining and safety < 200:
        safety += 1
        progressed = False
        for new_db in list(remaining):
            for edge in edges:
                db_from, db_to = _edge_db_names(edge, catalog)
                if not db_from or not db_to:
                    continue
                on_new_is_source = None
                anchor = None
                if new_db == db_from and db_to in connected:
                    on_new_is_source = True
                    anchor = db_to
                elif new_db == db_to and db_from in connected:
                    on_new_is_source = False
                    anchor = db_from
                else:
                    continue
                cond = join_cond(new_db, anchor, edge, on_new_is_source)
                if not cond:
                    continue
                join_lines.append(
                    f"LEFT JOIN `{new_db}` AS `{new_db}` ON {cond}"
                )
                connected.add(new_db)
                remaining.discard(new_db)
                progressed = True
                break
            if progressed:
                break
        if not progressed:
            break

    for t in sorted(remaining):
        join_lines.append(
            f"-- WARNING: no FK edge connects `{t}` to the anchored set; replace this placeholder JOIN."
        )
        join_lines.append(f"CROSS JOIN `{t}` AS `{t}`")
        notes.append(f"Missing FK path for table {t}; used CROSS JOIN placeholder.")

    return "\n".join(join_lines), notes
