import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_table_edges(metadata: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Convert extracted foreign keys into table dependency edges.

    Convention used:
    - If FK source_table -> target_table, then source_table *depends on* target_table.
      i.e. edge: source_table -> target_table
    """
    foreign_keys = metadata.get("foreign_keys") or []

    edges: List[Dict[str, Any]] = []
    adjacency: Dict[str, List[str]] = {}

    def add_edge(src: str, tgt: str, fk: Dict[str, Any]) -> None:
        if not src or not tgt:
            return
        adjacency.setdefault(src, [])
        if tgt not in adjacency[src]:
            adjacency[src].append(tgt)
        edges.append(
            {
                "from": src,
                "to": tgt,
                "foreign_key_id": fk.get("foreign_key_id") or fk.get("foreignKeyId"),
                "columns": [
                    {
                        "source_column_name": cm.get("source_column_name"),
                        "target_column_name": cm.get("target_column_name"),
                    }
                    for cm in (fk.get("column_mappings") or [])
                ],
            }
        )

    for fk in foreign_keys:
        src = fk.get("source_table_name") or fk.get("source_table_id") or ""
        tgt = fk.get("target_table_name") or fk.get("target_table_id") or ""
        add_edge(src, tgt, fk)

    return edges, adjacency


def to_dot(adjacency: Dict[str, List[str]], labels: Dict[str, str]) -> str:
    lines = ["digraph TABLE_DEPENDENCIES {", "  rankdir=LR;"]
    for node in sorted(adjacency.keys()):
        label = labels.get(node, node).replace('"', "'")
        lines.append(f'  "{node}" [label="{label}"];')
    for src, deps in adjacency.items():
        for dep in deps:
            if dep in adjacency or True:
                lines.append(f'  "{src}" -> "{dep}";')
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build table dependency graph from data_model_metadata.json")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/data_model_metadata.json",
        help="Path to data_model_metadata.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/table_dependency_graph",
        help="Output directory for graph artifacts",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(in_path)
    edges, adjacency = build_table_edges(metadata)

    # Labels from tables array (best-effort).
    labels: Dict[str, str] = {}
    for t in metadata.get("tables") or []:
        name = t.get("table_name")
        if name:
            labels[str(name)] = str(name)

    result = {
        "meta": {
            "table_count_in_metadata": len(metadata.get("tables") or []),
            "foreign_key_count": len(metadata.get("foreign_keys") or []),
            "edge_count": len(edges),
        },
        "edges": edges,
        "adjacency": adjacency,
    }

    json_out = out_dir / "table_dependency_graph.json"
    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    dot_out = out_dir / "table_dependency_graph.dot"
    dot_out.write_text(to_dot(adjacency, labels), encoding="utf-8")

    print(f"Wrote: {json_out}")
    print(f"Wrote: {dot_out}")


if __name__ == "__main__":
    main()

