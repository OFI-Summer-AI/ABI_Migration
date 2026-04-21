import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings

# Captures KPI ID from any KPI(...) call, including parameterised forms like KPI("id",{p1},{p2}).
# The closing ')' is NOT required so that extra parameters don't break the match.
KPI_CALL_RE = re.compile(r"""(?ix)
\bKPI\s*\(\s*
  (?:
    "([^"]+)"
    |'([^']+)'
  )
""")

# Matches a formula that is ENTIRELY a single KPI("id") call (no extra args).
DIRECT_KPI_REF_RE = re.compile(r"""(?ix)
^\s*KPI\s*\(\s*
  (?:
    "([^"]+)"
    |'([^']+)'
  )
\s*\)\s*$
""")

TABLE_COL_RE = re.compile(r'"([^"]+)"\."([^"]+)"')


class PQLExplainer:
    """Translate PQL formulas into plain-English business explanations."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=api_key,
        )
        self._cache: Dict[str, str] = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _record_usage(self, response: Any) -> None:
        md = getattr(response, "response_metadata", {}) or {}
        usage = md.get("token_usage") or getattr(response, "usage_metadata", {}) or {}
        in_t = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        out_t = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        tot_t = usage.get("total_tokens", in_t + out_t) or 0
        self.usage["input_tokens"] += int(in_t)
        self.usage["output_tokens"] += int(out_t)
        self.usage["total_tokens"] += int(tot_t)

    def explain(self, pql_formula: str, dependency_context: str = "") -> str:
        """Return a concise, plain-English explanation for one PQL formula."""
        normalized = (pql_formula or "").strip()
        if not normalized:
            return "No PQL formula available."

        cache_key = f"{normalized}||{dependency_context.strip()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        dependency_block = (
            f"Dependency context (resolved nested KPI formulas):\n{dependency_context}\n\n"
            if dependency_context
            else ""
        )

        messages = [
            SystemMessage(
                content=(
                    "You explain Celonis PQL formulas in plain English for business users. "
                    "Return exactly one sentence. "
                    "Be explicit about operation (sum/count/avg/etc), table, and column names. "
                    "Do not include markdown, bullets, or code fences. "
                    "Prefer semantic wording over mechanical wording: for indicator-style expressions "
                    "like SUM(CASE WHEN <condition> THEN 1 ELSE 0 END), describe them as "
                    "'count of rows/records meeting the condition' instead of 'sum of 1 and 0'. "
                    "More generally, when CASE emits indicator values (1/0, true/false flags), "
                    "explain the business intent (count/rate/share of matching records). "
                    "When the formula references {p1}, {p2} etc., treat those as user-defined "
                    "filter parameters applied at runtime."
                )
            ),
            HumanMessage(
                content=(
                    f'PQL: {normalized}\n\n'
                    f"{dependency_block}"
                    "Example style:\n"
                    'Input: SUM("Book"."PageCount")\n'
                    'Output: Calculate the total sum of the "PageCount" column from the "Book" table.\n'
                    "Input: SUM(CASE WHEN \"Sales\".\"IsValid\" = 'Y' THEN 1 ELSE 0 END)\n"
                    "Output: Count the number of records in the Sales table where IsValid equals 'Y'.\n\n"
                    "Now explain the given PQL."
                )
            ),
        ]

        response = self.llm.invoke(messages)
        self._record_usage(response)
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)

        explanation = str(content).strip().replace("\n", " ")
        explanation = clean_explanation_text(explanation)
        self._cache[cache_key] = explanation
        return explanation


def clean_explanation_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace('\\"', '"')
    cleaned = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def load_kpis(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("KPI JSON must be a list of objects")
    return [row for row in data if str(row.get("attribute_type", "")).lower() == "kpi"]


def safe_load_json(path: Path) -> Any:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def extract_kpi_refs(pql: str) -> List[str]:
    refs: List[str] = []
    for m in KPI_CALL_RE.finditer(pql or ""):
        ref = (m.group(1) or m.group(2) or "").strip()
        if ref:
            refs.append(ref)
    return refs


def extract_direct_nested_ref(pql: str) -> str:
    m = DIRECT_KPI_REF_RE.match(pql or "")
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").strip()


def build_kpi_index(kpis: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in kpis:
        if str(row.get("attribute_type", "")).lower() != "kpi":
            continue
        kpi_id = str(row.get("kpi_id", "") or "").strip()
        if kpi_id:
            index[kpi_id] = row
    return index


def build_graph(kpi_index: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {}
    for kpi_id, row in kpi_index.items():
        refs = [r for r in extract_kpi_refs(str(row.get("pql_formula", "") or "")) if r in kpi_index]
        seen: Set[str] = set()
        deps: List[str] = []
        for r in refs:
            if r not in seen:
                seen.add(r)
                deps.append(r)
        graph[kpi_id] = deps
    return graph


def topo_sort_bottom_up(graph: Dict[str, List[str]]) -> List[str]:
    """Return KPI IDs in leaf-first topological order (dependencies before dependents).
    Handles cycles by processing remaining nodes when stuck."""
    remaining = {n: set(deps) for n, deps in graph.items()}
    order: List[str] = []

    while remaining:
        ready = sorted([n for n, deps in remaining.items() if not deps])
        if not ready:
            # Cycle: break it by picking the node with fewest remaining deps
            ready = [min(remaining, key=lambda n: len(remaining[n]))]
        for n in ready:
            order.append(n)
            del remaining[n]
            for deps in remaining.values():
                deps.discard(n)

    return order


def resolve_formula_with_dependencies(
    kpi_id: str,
    graph: Dict[str, List[str]],
    kpi_index: Dict[str, Dict[str, Any]],
    visiting: Set[str],
) -> str:
    if kpi_id in visiting:
        return f"/* CYCLE:{kpi_id} */ NULL"
    row = kpi_index.get(kpi_id)
    if not row:
        return f"/* UNKNOWN_KPI:{kpi_id} */ NULL"

    formula = str(row.get("pql_formula", "") or "")
    visiting2 = set(visiting)
    visiting2.add(kpi_id)

    def repl(match: re.Match) -> str:
        ref = (match.group(1) or match.group(2) or "").strip()
        if not ref or ref not in kpi_index:
            return f'KPI("{ref}")'
        resolved = resolve_formula_with_dependencies(ref, graph, kpi_index, visiting2)
        return f"({resolved})"

    return KPI_CALL_RE.sub(repl, formula)


def build_dependency_paths(start: str, graph: Dict[str, List[str]]) -> List[str]:
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
    return [" -> ".join(p) for p in paths]


def extract_tables_from_pql(pql: str) -> List[str]:
    return sorted({t for t, _ in TABLE_COL_RE.findall(pql or "")})


def clean_prompt_context(text: str) -> str:
    cleaned = (text or "").replace("\n\n\n", "\n\n")
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\[\[.*?\]\]", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def build_rich_dependency_context(
    *,
    kpi_id: str,
    pql_formula: str,
    dependency_paths: List[str],
    expanded_formula: str,
    kpi_dependency_graph: Any,
    table_dependency_graph: Any,
    data_model_metadata: Any,
) -> str:
    parts: List[str] = []
    if dependency_paths:
        parts.append("Dependency paths:\n- " + "\n- ".join(dependency_paths))
    if expanded_formula:
        parts.append(f"Expanded PQL (KPI references inlined):\n{expanded_formula}")

    if isinstance(kpi_dependency_graph, dict):
        chains = kpi_dependency_graph.get("nested_kpi_chains") or []
        chain = next((c for c in chains if c.get("kpi_id") == kpi_id), None)
        if chain:
            parts.append(
                "KPI graph excerpt:\n"
                f"direct_dependencies: {chain.get('direct_dependencies')}\n"
                f"direct_dependents: {chain.get('direct_dependents')}"
            )

    ref_tables = set(extract_tables_from_pql(expanded_formula or pql_formula))
    if isinstance(data_model_metadata, dict) and ref_tables:
        fk_lines: List[str] = []
        for fk in data_model_metadata.get("foreign_keys") or []:
            s = fk.get("source_table_name")
            t = fk.get("target_table_name")
            if s in ref_tables and t in ref_tables:
                fk_lines.append(f"{s}->{t}")
        if fk_lines:
            parts.append("Relevant foreign keys: " + ", ".join(fk_lines[:20]))

    if isinstance(table_dependency_graph, dict) and ref_tables:
        edges = table_dependency_graph.get("edges") or []
        edge_lines = [
            f"{e.get('from')}->{e.get('to')}"
            for e in edges
            if e.get("from") in ref_tables and e.get("to") in ref_tables
        ]
        if edge_lines:
            parts.append("Relevant table dependency edges: " + ", ".join(edge_lines[:20]))

    return clean_prompt_context("\n\n".join(parts))[:6000]


def save_kpis(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate KPI PQL formulas to plain English.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(project_root / "data" / "raw" / "extracted_kpis_only.json"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "data" / "raw" / "extracted_kpis_explained.json"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env before running.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    kpis = load_kpis(input_path)
    kpi_index = build_kpi_index(kpis)
    graph = build_graph(kpi_index)

    # Process in topological order so dependency context is fully resolved
    # before we explain a KPI that depends on others.
    topo_order = topo_sort_bottom_up(graph)
    kpi_order_map = {kpi_id: i for i, kpi_id in enumerate(topo_order)}
    kpis_sorted = sorted(
        kpis,
        key=lambda row: kpi_order_map.get(str(row.get("kpi_id", "") or ""), 999999),
    )

    kpi_dep = safe_load_json(
        project_root / "data" / "processed" / "kpi_dependency_graph" / "kpi_dependency_graph.json"
    )
    table_dep = safe_load_json(
        project_root / "data" / "processed" / "table_dependency_graph" / "table_dependency_graph.json"
    )
    dm_meta = safe_load_json(project_root / "data" / "raw" / "data_model_metadata.json")

    explainer = PQLExplainer(api_key=api_key, model_name=args.model)

    for idx, kpi in enumerate(kpis_sorted, start=1):
        pql_formula = str(kpi.get("pql_formula", "") or "")
        kpi_id = str(kpi.get("kpi_id", "") or "")

        # Expand for ALL KPIs that have any dependencies (not just pure-delegation ones).
        deps = graph.get(kpi_id, [])
        paths: List[str] = []
        expanded = ""
        if deps:
            paths = build_dependency_paths(kpi_id, graph)
            expanded = resolve_formula_with_dependencies(kpi_id, graph, kpi_index, set())
            kpi["dependency_paths"] = paths
            kpi["pql_formula_expanded"] = expanded

        dependency_context = build_rich_dependency_context(
            kpi_id=kpi_id,
            pql_formula=pql_formula,
            dependency_paths=paths,
            expanded_formula=expanded,
            kpi_dependency_graph=kpi_dep,
            table_dependency_graph=table_dep,
            data_model_metadata=dm_meta,
        )
        try:
            kpi["pql_explanation"] = explainer.explain(pql_formula, dependency_context=dependency_context)
        except Exception as exc:
            kpi["pql_explanation"] = f"Explanation unavailable: {exc}"

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(kpis_sorted)} KPIs...")

    # Restore original order for output (sorted by original list position).
    original_order = {str(row.get("kpi_id", "")): i for i, row in enumerate(kpis)}
    kpis_sorted.sort(key=lambda row: original_order.get(str(row.get("kpi_id", "") or ""), 999999))

    save_kpis(output_path, kpis_sorted)
    print(f"Saved explained KPIs to: {output_path}")
    print(
        "LLM token usage (semantic translation): "
        f"input={explainer.usage['input_tokens']}, "
        f"output={explainer.usage['output_tokens']}, "
        f"total={explainer.usage['total_tokens']}"
    )


if __name__ == "__main__":
    main()
