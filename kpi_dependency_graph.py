import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


KPI_CALL_RE = re.compile(r"""(?ix)
\bKPI\s*\(\s*
  (?:                          # argument can be:
    "([^"]+)"                  # - double-quoted id
    |'([^']+)'                 # - single-quoted id
  )
\s*\)
""")

DIRECT_KPI_REF_RE = re.compile(r"""(?ix)
^\s*KPI\s*\(\s*
  (?:
    "([^"]+)"
    |'([^']+)'
  )
\s*\)\s*$
""")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list in extracted_kpis.json")
    return data


def extract_kpi_refs(pql: str) -> List[str]:
    """Return list of KPI ids referenced by KPI("...") calls in a PQL string."""
    refs: List[str] = []
    for m in KPI_CALL_RE.finditer(pql or ""):
        ref = m.group(1) or m.group(2) or ""
        ref = ref.strip()
        if ref:
            refs.append(ref)
    return refs


def extract_direct_nested_kpi_ref(pql: str) -> str:
    """Return referenced KPI id only when full formula is exactly KPI('...')."""
    m = DIRECT_KPI_REF_RE.match(pql or "")
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").strip()


def build_dependency_graph(kpi_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Build adjacency list graph: kpi_id -> [depends_on_kpi_ids...]
    Only includes rows where attribute_type == "KPI" (case-insensitive).
    """
    kpi_index: Dict[str, Dict[str, Any]] = {}
    for row in kpi_rows:
        if str(row.get("attribute_type", "")).lower() != "kpi":
            continue
        kpi_id = str(row.get("kpi_id", "") or "").strip()
        if not kpi_id:
            continue
        kpi_index[kpi_id] = row

    graph: Dict[str, List[str]] = {}
    for kpi_id, row in kpi_index.items():
        pql = str(row.get("pql_formula", "") or "")
        refs = extract_kpi_refs(pql)
        # Keep only references that point to KPIs we know about (avoid noise).
        refs_known = [r for r in refs if r in kpi_index]
        # De-dup while preserving order.
        seen: Set[str] = set()
        deps: List[str] = []
        for r in refs_known:
            if r not in seen:
                seen.add(r)
                deps.append(r)
        graph[kpi_id] = deps

    return graph, kpi_index


def filter_to_dependency_chains(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Keep only nodes that are part of at least one dependency chain.
    - Start nodes: KPIs with outgoing edges.
    - Include all reachable dependencies from those starts.
    """
    starts = [k for k, deps in graph.items() if deps]
    if not starts:
        return {}

    keep: Set[str] = set()
    stack = list(starts)
    while stack:
        node = stack.pop()
        if node in keep:
            continue
        keep.add(node)
        for dep in graph.get(node, []):
            if dep not in keep:
                stack.append(dep)

    return {k: [d for d in graph.get(k, []) if d in keep] for k in keep}


def build_paths_from(node: str, graph: Dict[str, List[str]], path: List[str], out: List[List[str]], seen: Set[str]) -> None:
    if node in seen:
        out.append(path + [f"[CYCLE:{node}]"])
        return
    deps = graph.get(node, [])
    if not deps:
        out.append(path + [node])
        return
    seen2 = set(seen)
    seen2.add(node)
    for dep in deps:
        build_paths_from(dep, graph, path + [node], out, seen2)


def build_paths_as_arrays(start: str, graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Return all dependency paths starting at `start`.
    Each path is represented as a list of KPI ids: [start, dep1, dep2, ...].
    """
    paths: List[List[str]] = []

    def dfs(node: str, path: List[str], seen: Set[str]) -> None:
        if node in seen:
            paths.append(path + [f"[CYCLE:{node}]"])
            return
        deps = graph.get(node, [])
        if not deps:
            paths.append(path + [node])
            return
        seen2 = set(seen)
        seen2.add(node)
        for dep in deps:
            dfs(dep, path + [node], seen2)

    dfs(start, [], set())
    return paths


def reachable_deps_in_order(start: str, graph: Dict[str, List[str]]) -> List[str]:
    """
    Compute transitive dependencies for `start` (excluding `start`),
    preserving a stable first-seen order.
    """
    seen: Set[str] = set()
    order: List[str] = []
    stack = [start]
    while stack:
        node = stack.pop()
        for dep in graph.get(node, []):
            if dep in seen:
                continue
            seen.add(dep)
            order.append(dep)
            stack.append(dep)
    return order


def stable_name(kpi_index: Dict[str, Dict[str, Any]], kpi_id: str) -> str:
    row = kpi_index.get(kpi_id, {}) or {}
    name = row.get("name")
    if name is None:
        return kpi_id
    name = str(name).strip()
    return name or kpi_id


def invert_graph(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Reverse adjacency: kpi_id -> [kpis_that_depend_on_it...]"""
    rev: Dict[str, List[str]] = defaultdict(list)
    for src, deps in graph.items():
        for dep in deps:
            rev[dep].append(src)
    return dict(rev)


def topo_sort_or_cycles(graph: Dict[str, List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Kahn topological sort. Returns (order, cycles).
    If cycles exist, order will contain only the acyclic portion.
    cycles is a best-effort list of strongly-connected components > 1 node
    (implemented via DFS stack for simplicity).
    """
    indeg: Dict[str, int] = {n: 0 for n in graph}
    for n, deps in graph.items():
        for d in deps:
            if d in indeg:
                indeg[n] += 1

    q = deque([n for n, d in indeg.items() if d == 0])
    order: List[str] = []
    deps_map = {n: list(deps) for n, deps in graph.items()}

    while q:
        n = q.popleft()
        order.append(n)
        for child, deps in deps_map.items():
            if n in deps:
                indeg[child] -= 1
                deps.remove(n)
                if indeg[child] == 0:
                    q.append(child)

    remaining = [n for n, d in indeg.items() if d > 0]
    if not remaining:
        return order, []

    # Best-effort cycle grouping: DFS to find SCCs among remaining nodes.
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    cycles: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in graph.get(v, []):
            if w not in graph:
                continue
            if w not in remaining:
                continue
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                cycles.append(sorted(comp))

    for v in remaining:
        if v not in idx:
            strongconnect(v)

    # Dedup SCCs
    uniq = []
    seen = set()
    for c in cycles:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    return order, uniq


def to_dot(graph: Dict[str, List[str]], labels: Dict[str, str]) -> str:
    lines = ["digraph KPI_DEPENDENCIES {", "  rankdir=LR;"]
    for node in sorted(graph.keys()):
        label = labels.get(node, node).replace('"', "'")
        lines.append(f'  "{node}" [label="{label}"];')
    for src, deps in graph.items():
        for dep in deps:
            if dep in graph:
                lines.append(f'  "{src}" -> "{dep}";')
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KPI dependency graph from extracted_kpis.json")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/extracted_kpis.json",
        help="Path to extracted_kpis.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/kpi_dependency_graph",
        help="Output directory for graph artifacts",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(in_path)
    graph, kpi_index = build_dependency_graph(rows)
    chain_graph = filter_to_dependency_chains(graph)
    rev = invert_graph(chain_graph)

    nested_kpis = sorted(
        [
            kpi_id
            for kpi_id, row in kpi_index.items()
            if extract_direct_nested_kpi_ref(str(row.get("pql_formula", "") or ""))
        ]
    )

    order, cycles = topo_sort_or_cycles(chain_graph)

    # Helpful labels: "KPI Name (kpi_id)" for DOT output only.
    labels = {k: f"{stable_name(kpi_index, k)} ({k})".strip() for k in chain_graph.keys()}

    # Nodes with no dependencies and nodes depended-on by others
    roots = sorted([k for k, deps in chain_graph.items() if not deps])
    leaves = sorted([k for k in chain_graph.keys() if k not in rev])

    # Human-friendly output objects
    direct_dependencies = {k: chain_graph.get(k, []) for k in chain_graph.keys()}
    direct_dependents = {k: rev.get(k, []) for k in chain_graph.keys()}

    nested_kpi_chains: List[Dict[str, Any]] = []
    for nk in nested_kpis:
        if nk not in chain_graph:
            continue

        direct_dep = chain_graph.get(nk, [])
        transitive = reachable_deps_in_order(nk, chain_graph)
        paths_arrays = build_paths_as_arrays(nk, chain_graph)
        paths_strings = [" -> ".join(p) for p in (paths_arrays or [])]

        nested_kpi_chains.append(
            {
                "kpi_id": nk,
                "kpi_name": stable_name(kpi_index, nk),
                "definition_pql": str(kpi_index.get(nk, {}).get("pql_formula", "") or "").strip(),
                "direct_dependencies": direct_dep,
                "transitive_dependencies": transitive,
                "dependency_paths": paths_strings,
            }
        )

    edges_list: List[Dict[str, str]] = []
    for src, deps in chain_graph.items():
        for dep in deps:
            edges_list.append({"from": src, "to": dep})

    summary = {
        "meta": {
            "total_kpis_in_input": len(graph),
            "total_kpis_in_dependency_chains": len(chain_graph),
            "total_edges_in_dependency_chains": sum(len(v) for v in chain_graph.values()),
            "nested_kpis_direct_format_count": len(nested_kpis),
        },
        "nested_kpi_chains": nested_kpi_chains,
        "direct_dependencies": direct_dependencies,
        "direct_dependents": direct_dependents,
        "edges": edges_list,
        "roots_no_dependencies": roots,
        "leaves_no_dependents": leaves,
        "topological_order_partial": order,
        "cycles": cycles,
    }

    (out_dir / "kpi_dependency_graph.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "kpi_dependency_graph.dot").write_text(
        to_dot(chain_graph, labels),
        encoding="utf-8",
    )

    print(f"Input KPIs: {summary['meta']['total_kpis_in_input']}")
    print(f"KPIs in dependency chains: {summary['meta']['total_kpis_in_dependency_chains']}")
    print(
        f"Edges in dependency chains: {summary['meta']['total_edges_in_dependency_chains']}"
    )
    print(f"Nested KPIs (direct KPI('...') format): {len(nested_kpis)}")
    print(f"Roots (no deps): {len(roots)}")
    print(f"Leaves (no dependents): {len(leaves)}")
    if cycles:
        print(f"Cycles detected: {len(cycles)}")
    else:
        print("No cycles detected.")
    print(f"Wrote: {out_dir / 'kpi_dependency_graph.json'}")
    print(f"Wrote: {out_dir / 'kpi_dependency_graph.dot'}")


if __name__ == "__main__":
    main()

