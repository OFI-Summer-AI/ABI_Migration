import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from langchain_openai import ChatOpenAI
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


TABLE_COL_RE = re.compile(r'"([^"]+)"\."([^"]+)"')


def safe_load_json(path: Path) -> Any:
    """Load JSON from path if it exists, otherwise return None."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def extract_tables_and_columns_from_pql(pql: str) -> Dict[str, Set[str]]:
    """
    Extract referenced tables/columns from PQL/SQL snippets in the form:
      "TABLE"."COLUMN"
    """
    refs: Dict[str, Set[str]] = {}
    for t, c in TABLE_COL_RE.findall(pql or ""):
        refs.setdefault(t, set()).add(c)
    return refs


def build_rich_dependency_context(
    kpi_id: str,
    dependency_context: str,
    expanded_pql: str,
    *,
    kpi_dependency_graph: Any,
    data_model_metadata: Any,
    table_dependency_graph: Any,
) -> str:
    """
    Build a compact context string for nested KPI translation.

    To avoid token overload, this function filters the large JSON artifacts down to the referenced tables/columns and the relevant edges.
    """
    parts: List[str] = []
    if dependency_context.strip():
        parts.append(dependency_context.strip())

    # 1) KPI dependency graph excerpt
    if isinstance(kpi_dependency_graph, dict):
        nested_chains = kpi_dependency_graph.get("nested_kpi_chains") or []
        chain = next((c for c in nested_chains if c.get("kpi_id") == kpi_id), None)
        if chain:
            parts.append(
                "KPI dependency graph excerpt:\n"
                f"kpi_name: {chain.get('kpi_name')}\n"
                f"direct_dependencies: {chain.get('direct_dependencies')}\n"
                f"transitive_dependencies_count: {len(chain.get('transitive_dependencies') or [])}\n"
            )
            paths = chain.get("dependency_paths") or []
            if paths:
                parts.append("dependency_paths (examples):\n- " + "\n- ".join(paths[:8]))

    # 2) Data model excerpt for referenced tables
    table_cols = extract_tables_and_columns_from_pql(expanded_pql)
    referenced_tables = sorted(table_cols.keys())

    if isinstance(data_model_metadata, dict) and referenced_tables:
        tables_index = {t.get("table_name"): t for t in (data_model_metadata.get("tables") or [])}
        fk_list = data_model_metadata.get("foreign_keys") or []

        # Columns: show only referenced columns if possible
        col_lines: List[str] = []
        for tbl in referenced_tables[:15]:
            wanted_cols = sorted(table_cols.get(tbl) or [])
            table_obj = tables_index.get(tbl) or {}
            all_cols = table_obj.get("columns") or []
            cols_to_show = wanted_cols
            if wanted_cols and all_cols:
                cols_to_show = [c for c in wanted_cols if c in all_cols]
            if wanted_cols and not cols_to_show:
                cols_to_show = wanted_cols
            if not cols_to_show and all_cols:
                cols_to_show = all_cols[:50]

            col_lines.append(f"- {tbl}: columns({len(cols_to_show)}): {cols_to_show[:30]}")

        if col_lines:
            parts.append("Data model excerpt (referenced tables/columns):\n" + "\n".join(col_lines))

        # Foreign keys among referenced tables
        fk_lines: List[str] = []
        ref_set = set(referenced_tables)
        for fk in fk_list:
            src = fk.get("source_table_name")
            tgt = fk.get("target_table_name")
            if src in ref_set and tgt in ref_set:
                mappings = fk.get("column_mappings") or []
                mapping_str = ", ".join(
                    f"{m.get('source_column_name')}->{m.get('target_column_name')}"
                    for m in mappings
                    if m.get("source_column_name") and m.get("target_column_name")
                )
                fk_lines.append(f"- {src} -> {tgt} ({mapping_str})".strip())

        if fk_lines:
            parts.append("Foreign key relations among referenced tables:\n" + "\n".join(fk_lines[:30]))

    # 3) Table dependency graph excerpt (filtered edge list)
    if isinstance(table_dependency_graph, dict):
        edges = table_dependency_graph.get("edges") or []
        ref_set = set(referenced_tables)
        filtered_edges = []
        for e in edges:
            if e.get("from") in ref_set and e.get("to") in ref_set:
                filtered_edges.append(e)
            if len(filtered_edges) >= 40:
                break
        if filtered_edges:
            parts.append(
                "Table dependency graph edges (filtered):\n"
                + "\n".join(f"- {e.get('from')} -> {e.get('to')}" for e in filtered_edges)
            )

    combined = "\n\n".join([p for p in parts if p])
    # Hard cap: reduce chance of prompt overflow
    return combined[:12000]


class PQLExplainer:
    """Translate PQL formulas into plain-English business explanations."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
        )
        self._cache: Dict[str, str] = {}

    def _extract_semantics(self, source_pql: str, dependency_context: str = "") -> Dict[str, Any]:
        """
        Step 1: Ask the LLM for structured semantic extraction in strict JSON.
        This reduces hallucination for nested KPI formulas.
        """
        dependency_block = (
            f"Dependency context (for nested KPI resolution):\n{dependency_context}\n\n"
            if dependency_context
            else ""
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a precise Celonis PQL semantic parser.\n"
                    "Return valid JSON only, no markdown.\n"
                    "Use the provided PQL as the source of truth; do not invent logic.\n"
                    "If nested KPI context exists, use it only to disambiguate.\n"
                    "Required JSON keys:\n"
                    "- operation_summary: string\n"
                    "- measures: array of strings\n"
                    "- filters: array of strings\n"
                    "- grouping_grain: string\n"
                    "- referenced_tables: array of strings\n"
                    "- referenced_columns: array of strings\n"
                    "- caveats: array of strings\n"
                )
            ),
            HumanMessage(
                content=(
                    f"PQL:\n{source_pql}\n\n"
                    f"{dependency_block}"
                    "Extract the semantics now."
                )
            ),
        ]

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        text = str(content).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Conservative fallback if JSON parse fails.
        return {
            "operation_summary": "Could not reliably parse structured semantics from model output.",
            "measures": [],
            "filters": [],
            "grouping_grain": "Unknown",
            "referenced_tables": [],
            "referenced_columns": [],
            "caveats": ["Model returned non-JSON output during semantic extraction."],
        }

    def _render_explanation(
        self,
        *,
        raw_pql: str,
        source_pql: str,
        structured_semantics: Dict[str, Any],
        is_nested: bool,
        dependency_context: str = "",
    ) -> str:
        """
        Step 2: Render a business-readable explanation from structured semantics.
        Allows 1-3 sentences for complex nested KPIs.
        """
        dependency_block = (
            f"Dependency context (for nested KPI resolution):\n{dependency_context}\n\n"
            if dependency_context
            else ""
        )

        messages = [
            SystemMessage(
                content=(
                    "You explain Celonis KPIs for business and analytics users.\n"
                    "Output plain text only (no markdown).\n"
                    "Write 1 to 3 concise sentences.\n"
                    "Be explicit about operation, key filters, and tables/columns when present.\n"
                    "For nested KPIs, treat expanded PQL as semantic source of truth.\n"
                    "Do not invent joins, tables, columns, or business rules.\n"
                    "If semantics are uncertain, state uncertainty briefly in the final sentence."
                )
            ),
            HumanMessage(
                content=(
                    f"Raw PQL:\n{raw_pql}\n\n"
                    f"Source PQL used for semantics:\n{source_pql}\n\n"
                    f"Is nested KPI: {is_nested}\n\n"
                    f"Structured semantics JSON:\n{json.dumps(structured_semantics, ensure_ascii=False)}\n\n"
                    f"{dependency_block}"
                    "Now produce the final explanation."
                )
            ),
        ]

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        return str(content).strip()

    def explain(self, pql_formula: str, dependency_context: str = "", expanded_pql: str = "") -> str:
        """Return a concise, plain-English explanation for one PQL formula."""
        normalized = (pql_formula or "").strip()
        if not normalized:
            return "No PQL formula available."
        normalized_expanded = (expanded_pql or "").strip()
        source_pql = normalized_expanded or normalized
        is_nested = bool(normalized_expanded and normalized_expanded != normalized)

        cache_key = f"{normalized}||{normalized_expanded}||{dependency_context.strip()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        semantics = self._extract_semantics(source_pql=source_pql, dependency_context=dependency_context)
        explanation = self._render_explanation(
            raw_pql=normalized,
            source_pql=source_pql,
            structured_semantics=semantics,
            is_nested=is_nested,
            dependency_context=dependency_context,
        ).replace("\n", " ")
        explanation = clean_explanation_text(explanation)
        explanation = normalize_indicator_aggregation_explanation(source_pql, explanation)
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


def normalize_indicator_aggregation_explanation(pql_formula: str, explanation: str) -> str:
    """
    General semantic normalization:
    For indicator aggregations like SUM(CASE WHEN <cond> THEN 1 ELSE 0 END),
    prefer business wording ("count of rows where <cond>") over mechanical wording
    ("sum of 1 else 0"). This applies broadly across many KPI definitions.
    """
    pql = (pql_formula or "").strip()
    out = (explanation or "").strip()
    if not pql or not out:
        return out

    indicator_sum_pattern = re.compile(
        r"""(?is)
        ^\s*SUM\s*\(\s*
        CASE\s+WHEN\s+(?P<cond>.+?)\s+
        THEN\s+1\s+ELSE\s+0\s+END
        \s*\)\s*$
        """,
        re.VERBOSE,
    )
    m = indicator_sum_pattern.match(pql)
    if not m:
        return out

    cond = re.sub(r"\s+", " ", m.group("cond")).strip()
    cond_english = pql_condition_to_english(cond)
    return (
        "Count the number of rows where "
        f"{cond_english}."
    )


def pql_condition_to_english(condition: str) -> str:
    """
    Convert simple PQL/SQL boolean conditions into readable English.
    This is intentionally lightweight and generic for common KPI conditions.
    """
    cond = (condition or "").strip()
    if not cond:
        return "the condition is true"

    # Normalize whitespace and unwrap excessive outer parentheses.
    cond = re.sub(r"\s+", " ", cond)
    while cond.startswith("(") and cond.endswith(")"):
        inner = cond[1:-1].strip()
        if not inner:
            break
        cond = inner

    # Replace identifiers like "Table"."Column" -> column Column in table Table
    cond = re.sub(
        r'"([^"]+)"\."([^"]+)"',
        lambda m: f'column {m.group(2)} in table {m.group(1)}',
        cond,
    )
    cond = re.sub(
        r'"([A-Za-z_][A-Za-z0-9_]*)"',
        lambda m: f'column {m.group(1)}',
        cond,
    )

    # Replace common operators/keywords with English.
    replacements = [
        (r"(?i)\bIS\s+NOT\s+NULL\b", "is not null"),
        (r"(?i)\bIS\s+NULL\b", "is null"),
        (r"<>", "is not equal to"),
        (r"!=", "is not equal to"),
        (r">=", "is greater than or equal to"),
        (r"<=", "is less than or equal to"),
        (r">", "is greater than"),
        (r"<", "is less than"),
        (r"=", "is equal to"),
        (r"(?i)\bAND\b", "and"),
        (r"(?i)\bOR\b", "or"),
        (r"(?i)\bIN\b", "in"),
        (r"(?i)\bNOT\b", "not"),
    ]
    for pattern, replacement in replacements:
        cond = re.sub(pattern, f" {replacement} ", cond)

    # Keep quoted literals but strip escape artifacts.
    cond = cond.replace('\\"', '"')
    cond = re.sub(r"\s{2,}", " ", cond).strip()
    return cond


def load_kpis(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("KPI JSON must be a list of objects")
    # Only KPIs are relevant for translation (exclude Filters and Attributes).
    return [row for row in data if str(row.get("attribute_type", "")).lower() == "kpi"]


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
        default="gpt-4o-mini",
        help="OpenAI chat model name (e.g. gpt-4o-mini, gpt-4o)",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY (or openai_api_key in .env) is required for KPI translation.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    kpis = load_kpis(input_path)
    explainer = PQLExplainer(api_key=api_key, model_name=args.model)
    kpi_index = build_kpi_index(kpis)
    graph = build_graph(kpi_index)

    # Optional context artifacts for nested KPI interpretation.
    # These are used to reduce hallucination and improve explanation accuracy.
    kpi_dependency_graph_path = project_root / "data" / "processed" / "kpi_dependency_graph" / "kpi_dependency_graph.json"
    table_dependency_graph_path = project_root / "data" / "processed" / "table_dependency_graph" / "table_dependency_graph.json"
    data_model_metadata_path = project_root / "data" / "raw" / "data_model_metadata.json"

    kpi_dependency_graph = safe_load_json(kpi_dependency_graph_path)
    table_dependency_graph = safe_load_json(table_dependency_graph_path)
    data_model_metadata = safe_load_json(data_model_metadata_path)

    for idx, kpi in enumerate(kpis, start=1):
        pql_formula = str(kpi.get("pql_formula", "") or "")
        kpi_id = str(kpi.get("kpi_id", "") or "")
        direct_nested_ref = extract_direct_nested_ref(pql_formula)

        dependency_context = ""
        expanded = ""

        referenced_kpis = extract_kpi_refs(pql_formula)
        if referenced_kpis and kpi_id in graph:
            paths = build_dependency_paths(kpi_id, graph)
            expanded = resolve_formula_with_dependencies(kpi_id, graph, kpi_index, set())
            dependency_context = (
                f"Dependency paths (KPI expansion chain):\n- " + "\n- ".join(paths) + "\n\n"
                f"Expanded PQL by dependency substitution:\n{expanded}"
            )
            kpi["dependency_paths"] = paths
            kpi["pql_formula_expanded"] = expanded

            # Add additional context from dependency/table/data-model artifacts.
            if dependency_context.strip() and expanded:
                dependency_context = build_rich_dependency_context(
                    kpi_id,
                    dependency_context=dependency_context,
                    expanded_pql=expanded,
                    kpi_dependency_graph=kpi_dependency_graph,
                    data_model_metadata=data_model_metadata,
                    table_dependency_graph=table_dependency_graph,
                )
        try:
            kpi["pql_explanation"] = explainer.explain(
                pql_formula,
                dependency_context=dependency_context,
                expanded_pql=expanded,
            )
        except Exception as exc:
            # Keep processing remaining KPIs; attach a fallback message per failed row.
            kpi["pql_explanation"] = f"Explanation unavailable: {exc}"
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(kpis)} KPIs...")

    save_kpis(output_path, kpis)
    print(f"Saved explained KPIs to: {output_path}")


if __name__ == "__main__":
    main()
