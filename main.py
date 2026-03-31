import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_by_kpi_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        kpi_id = str(row.get("kpi_id", "") or "").strip()
        if kpi_id:
            out[kpi_id] = row
    return out


def fmt_list(values: List[str], default: str = "None") -> str:
    if not values:
        return default
    return ", ".join(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show KPI PQL -> English -> SQL with dependency context."
    )
    parser.add_argument(
        "--explained",
        type=str,
        default="data/raw/extracted_kpis_explained.json",
        help="Path to explained KPI JSON",
    )
    parser.add_argument(
        "--sql",
        type=str,
        default="data/translation/kpis_sql_translation.json",
        help="Path to KPI SQL translation JSON",
    )
    parser.add_argument(
        "--deps",
        type=str,
        default="data/processed/kpi_dependency_graph/kpi_dependency_graph.json",
        help="Path to KPI dependency graph JSON",
    )
    parser.add_argument(
        "--kpi-id",
        type=str,
        default="",
        help="Optional: print only one KPI by id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: limit number of KPIs shown (0 = all)",
    )
    args = parser.parse_args()

    explained_path = Path(args.explained)
    sql_path = Path(args.sql)
    deps_path = Path(args.deps)

    if not explained_path.exists():
        raise FileNotFoundError(f"Missing explained KPI file: {explained_path}")
    if not sql_path.exists():
        raise FileNotFoundError(f"Missing SQL translation file: {sql_path}")
    if not deps_path.exists():
        raise FileNotFoundError(f"Missing KPI dependency graph file: {deps_path}")

    explained_rows = load_json(explained_path)
    sql_rows = load_json(sql_path)
    deps = load_json(deps_path)

    if not isinstance(explained_rows, list):
        raise ValueError("Explained KPI file must be a JSON list")
    if not isinstance(sql_rows, list):
        raise ValueError("SQL translation file must be a JSON list")
    if not isinstance(deps, dict):
        raise ValueError("Dependency graph file must be a JSON object")

    explained_idx = index_by_kpi_id(explained_rows)
    sql_idx = index_by_kpi_id(sql_rows)

    direct_deps = deps.get("direct_dependencies") or {}
    direct_dependents = deps.get("direct_dependents") or {}
    nested_chains = deps.get("nested_kpi_chains") or []
    nested_by_id = {str(item.get("kpi_id", "")): item for item in nested_chains}

    all_kpi_ids = sorted(set(explained_idx.keys()) | set(sql_idx.keys()))
    if args.kpi_id:
        all_kpi_ids = [k for k in all_kpi_ids if k == args.kpi_id]
    if args.limit and args.limit > 0:
        all_kpi_ids = all_kpi_ids[: args.limit]

    print("=" * 100)
    print("KPI TRANSLATION REPORT (PQL -> ENGLISH -> SQL + DEPENDENCIES)")
    print("=" * 100)
    print(f"Total KPIs available: {len(set(explained_idx.keys()) | set(sql_idx.keys()))}")
    print(f"KPIs shown: {len(all_kpi_ids)}")
    if args.kpi_id:
        print(f"Filter: kpi_id={args.kpi_id}")
    print("-" * 100)

    for i, kpi_id in enumerate(all_kpi_ids, start=1):
        e = explained_idx.get(kpi_id, {})
        s = sql_idx.get(kpi_id, {})

        name = e.get("name") or s.get("name") or ""
        pql = e.get("pql_formula") or s.get("pql_formula") or ""
        english = e.get("pql_explanation") or s.get("pql_explanation") or ""
        sql = s.get("sql_query") or ""
        sql_status = s.get("sql_translation_status") or "not_translated"

        deps_list = direct_deps.get(kpi_id, [])
        dependents_list = direct_dependents.get(kpi_id, [])
        nested = nested_by_id.get(kpi_id, {})
        dep_paths = nested.get("dependency_paths", []) if isinstance(nested, dict) else []

        print(f"[{i}] KPI: {name}")
        print(f"ID: {kpi_id}")
        print("-" * 100)
        print("PQL:")
        print(pql if pql else "N/A")
        print()
        print("PLAIN ENGLISH:")
        print(english if english else "N/A")
        print()
        print(f"SQL (status: {sql_status}):")
        print(sql if sql else "N/A")
        print()
        print("DEPENDENCIES:")
        print(f"- Direct dependencies: {fmt_list(deps_list)}")
        print(f"- Direct dependents: {fmt_list(dependents_list)}")
        if dep_paths:
            print("- Dependency paths:")
            for p in dep_paths:
                print(f"  * {p}")
        else:
            print("- Dependency paths: None")
        print("=" * 100)


if __name__ == "__main__":
    main()

