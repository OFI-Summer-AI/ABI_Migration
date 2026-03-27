import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings

KPI_CALL_RE = re.compile(r"""(?ix)
\bKPI\s*\(\s*
  (?:
    "([^"]+)"
    |'([^']+)'
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


class PQLExplainer:
    """Translate PQL formulas into plain-English business explanations."""

    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-20b"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0,
            groq_api_key=api_key,
        )
        self._cache: Dict[str, str] = {}

    def explain(self, pql_formula: str, dependency_context: str = "") -> str:
        """Return a concise, plain-English explanation for one PQL formula."""
        normalized = (pql_formula or "").strip()
        if not normalized:
            return "No PQL formula available."

        cache_key = f"{normalized}||{dependency_context.strip()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        dependency_block = (
            f"Dependency context (for nested KPI resolution):\n{dependency_context}\n\n"
            if dependency_context
            else ""
        )

        messages = [
            SystemMessage(
                content=(
                    "You explain Celonis PQL formulas in plain English for business users. "
                    "Return exactly one sentence. "
                    "Be explicit about operation (sum/count/avg/etc), table, and column names. "
                    "Do not include markdown, bullets, or code fences."
                )
            ),
            HumanMessage(
                content=(
                    f'PQL: {normalized}\n\n'
                    f"{dependency_block}"
                    "Example style:\n"
                    'Input: SUM("Book"."PageCount")\n'
                    'Output: Calculate the total sum of the "PageCount" column from the "Book" table.\n\n'
                    "Now explain the given PQL."
                )
            ),
        ]

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)

        explanation = str(content).strip().replace("\n", " ")
        explanation = clean_explanation_text(explanation)
        self._cache[cache_key] = explanation
        return explanation


def clean_explanation_text(text: str) -> str:
    """
    Remove escaped/quoted identifier artifacts from natural-language output.
    This only cleans the explanation text; it does NOT change source PQL/SQL.
    """
    cleaned = text.strip()

    # Convert escaped quotes to normal quotes.
    cleaned = cleaned.replace('\\"', '"')

    # Remove quotes around identifier-like names (table/column names),
    # e.g. "o_custom_SalesOrder" -> o_custom_SalesOrder.
    cleaned = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r"\1", cleaned)

    # Normalize repeated spaces produced by replacements.
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def load_kpis(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("KPI JSON must be a list of objects")
    return data


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
            return f"KPI(\"{ref}\")"
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


def save_kpis(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate KPI PQL formulas to plain English.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(project_root / "data" / "raw" / "extracted_kpis.json"),
        help="Path to input KPI JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "data" / "raw" / "extracted_kpis_explained.json"),
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Groq model name",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY (or groq_api_key in .env) is required for KPI translation.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    kpis = load_kpis(input_path)
    explainer = PQLExplainer(api_key=api_key, model_name=args.model)
    kpi_index = build_kpi_index(kpis)
    graph = build_graph(kpi_index)

    for idx, kpi in enumerate(kpis, start=1):
        pql_formula = str(kpi.get("pql_formula", "") or "")
        kpi_id = str(kpi.get("kpi_id", "") or "")
        direct_nested_ref = extract_direct_nested_ref(pql_formula)

        dependency_context = ""
        if direct_nested_ref and kpi_id in graph:
            paths = build_dependency_paths(kpi_id, graph)
            expanded = resolve_formula_with_dependencies(kpi_id, graph, kpi_index, set())
            dependency_context = (
                f"Nested KPI id: {kpi_id}\n"
                f"Direct dependency: {direct_nested_ref}\n"
                f"Dependency paths:\n- " + "\n- ".join(paths) + "\n\n"
                f"Expanded PQL by dependency substitution:\n{expanded}"
            )
            kpi["dependency_paths"] = paths
            kpi["pql_formula_expanded"] = expanded
        try:
            kpi["pql_explanation"] = explainer.explain(pql_formula, dependency_context=dependency_context)
        except Exception as exc:
            # Keep processing remaining KPIs; attach a fallback message per failed row.
            kpi["pql_explanation"] = f"Explanation unavailable: {exc}"
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(kpis)} KPIs...")

    save_kpis(output_path, kpis)
    print(f"Saved explained KPIs to: {output_path}")


if __name__ == "__main__":
    main()
