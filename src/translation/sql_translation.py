import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings

KPI_CALL_RE = re.compile(r"""(?ix)\bKPI\s*\(\s*(?:"([^"]+)"|'([^']+)')\s*\)""")
TABLE_COL_RE = re.compile(r'"([^"]+)"\."([^"]+)"')


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of KPI objects.")
    return [row for row in data if str(row.get("attribute_type", "")).lower() == "kpi"]


def safe_load_json(path: Path) -> Any:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def save_json_rows(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def save_pairs_report_txt(path: Path, pairs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for i, pair in enumerate(pairs, start=1):
        kpi_id = pair.get("kpi_id") or ""
        name = pair.get("name") or ""
        lines.append(f"[{i}] KPI_ID: {kpi_id} | NAME: {name}".rstrip())
        lines.append("PQL:")
        lines.append(str(pair.get("pql_formula", "") or ""))
        lines.append("SQL:")
        lines.append(str(pair.get("sql_query", "") or ""))
        lines.append("-" * 60)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def clean_english_explanation(text: str) -> str:
    """Clean noisy artifacts before SQL generation prompt."""
    cleaned = (text or "").strip()
    cleaned = cleaned.replace('\\"', '"')
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)  # remove unresolved placeholders like {p1}
    cleaned = re.sub(r"\[\[.*?\]\]", " ", cleaned)  # remove bracketed diagnostics/noise
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned)
    cleaned = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def extract_kpi_refs(pql: str) -> List[str]:
    refs: List[str] = []
    for m in KPI_CALL_RE.finditer(pql or ""):
        ref = (m.group(1) or m.group(2) or "").strip()
        if ref:
            refs.append(ref)
    return refs


def extract_tables_from_pql(pql: str) -> List[str]:
    return sorted({t for t, _ in TABLE_COL_RE.findall(pql or "")})


def build_sql_context(
    *,
    kpi_id: str,
    pql_formula: str,
    kpi_dependency_graph: Any,
    table_dependency_graph: Any,
    data_model_metadata: Any,
) -> str:
    parts: List[str] = []

    refs = extract_kpi_refs(pql_formula)
    if refs:
        parts.append(f"KPI references in formula: {refs}")

    if isinstance(kpi_dependency_graph, dict):
        chains = kpi_dependency_graph.get("nested_kpi_chains") or []
        chain = next((c for c in chains if c.get("kpi_id") == kpi_id), None)
        if chain:
            parts.append(
                "KPI dependency graph excerpt:\n"
                f"direct_dependencies: {chain.get('direct_dependencies')}\n"
                f"transitive_dependencies_count: {len(chain.get('transitive_dependencies') or [])}"
            )

    ref_tables = set(extract_tables_from_pql(pql_formula))
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
        edge_lines = [f"{e.get('from')}->{e.get('to')}" for e in edges if e.get("from") in ref_tables and e.get("to") in ref_tables]
        if edge_lines:
            parts.append("Relevant table dependency edges: " + ", ".join(edge_lines[:20]))

    return "\n".join(parts)[:6000]


class SQLTranslator:
    """Translate plain-English KPI explanations into SQL queries."""

    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-20b"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0,
            groq_api_key=api_key,
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

    def to_sql(self, explanation: str, pql_formula: str, extra_context: str = "") -> str:
        normalized_explanation = clean_english_explanation(explanation)
        normalized_pql = (pql_formula or "").strip()
        if not normalized_explanation and not normalized_pql:
            return "-- Missing explanation and PQL; SQL cannot be generated."

        cache_key = f"{normalized_pql}||{normalized_explanation}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        extra_block = (
            f"Additional dependency/table/model context:\n{extra_context}\n\n"
            if extra_context
            else ""
        )

        messages = [
            SystemMessage(
                content=(
                    # Generalized system prompt for PQL -> SQL translation.
                    #
                    # This prompt is intentionally domain-agnostic so it can translate any PQL
                    # expression (not only KPI examples). It focuses on semantic equivalence,
                    # deterministic output, and safe fallbacks for unsupported constructs.
                    "ROLE\n"
                    "- You are a precise transpiler that converts Celonis PQL expressions into SQL.\n"
                    "- Treat PQL as the single source of truth for logic.\n"
                    "- Use the English description only as supporting context when PQL is ambiguous.\n"
                    "\n"
                    "PRIMARY OBJECTIVE\n"
                    "- Produce SQL that preserves the exact business semantics of the given PQL.\n"
                    "- Do not optimize, simplify, or reinterpret intent.\n"
                    "\n"
                    "MANDATORY OUTPUT CONTRACT\n"
                    "1) Return exactly one SQL SELECT statement.\n"
                    "2) Return plain SQL text only (no markdown, no comments, no explanation).\n"
                    "3) Do not include a trailing semicolon.\n"
                    "4) Do not emit DDL or DML (no CREATE/ALTER/DROP/INSERT/UPDATE/DELETE/MERGE).\n"
                    "\n"
                    "SEMANTIC PRESERVATION RULES\n"
                    "- Preserve arithmetic, constants, operator precedence, and CASE logic exactly.\n"
                    "- Preserve DISTINCT semantics exactly where used.\n"
                    "- Preserve null behavior; do not introduce COALESCE/IFNULL unless PQL implies it.\n"
                    "- Preserve filter semantics; do not add or remove predicates.\n"
                    "- Preserve aggregation grain; do not add GROUP BY unless required by translation.\n"
                    "- Do not invent joins, tables, columns, or business rules not inferable from PQL.\n"
                    "\n"
                    "IDENTIFIERS, TYPES, AND DIALECT RULES\n"
                    "- Use  -style double quotes for identifiers; never use backticks.\n"
                    "- Keep source identifier names/case faithful to PQL.\n"
                    "- Add CAST only when needed to mirror explicit type intent in PQL.\n"
                    "- Example mapping: TO_FLOAT(x) -> CAST(x AS FLOAT).\n"
                    "- Do not add arbitrary precision/rounding (e.g., DECIMAL(10,2), ROUND) unless requested.\n"
                    "\n"
                    "FUNCTION/CONSTRUCT MAPPING GUIDELINES\n"
                    "- Map PQL aggregates and scalar functions to nearest SQL equivalents.\n"
                    "- Translate conditional constructs to CASE WHEN ... THEN ... ELSE ... END.\n"
                    "- Translate date/time logic conservatively; prefer explicit expressions over assumptions.\n"
                    "- For nested constructs, keep nesting and evaluation order equivalent to PQL.\n"
                    "\n"
                    "UNRESOLVED REFERENCES AND UNSUPPORTED FEATURES\n"
                    "- If a construct cannot be faithfully translated without missing context,\n"
                    "  still return a valid SELECT with NULL AS value while preserving any known context.\n"
                    "- Never fabricate missing logic to make the query look complete.\n"
                    "\n"
                    "QUALITY CHECK BEFORE RETURN\n"
                    "- Ensure output is syntactically valid SQL SELECT text.\n"
                    "- Ensure every retained expression is traceable to the provided PQL.\n"
                )
            ),
            HumanMessage(
                content=(
                    f"KPI plain-English explanation (may omit technical details):\n"
                    f"{normalized_explanation}\n\n"
                    f"Original PQL (source of truth for semantics):\n"
                    f"{normalized_pql}\n\n"
                    f"{extra_block}"
                    f"Generate the equivalent SQL SELECT query for this KPI."
                )
            ),
        ]

        response = self.llm.invoke(messages)
        self._record_usage(response)
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)

        sql = str(content).strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)

        # Normalize any model-produced identifier quoting to  style.
        sql = re.sub(r"`([^`]+)`", r'"\1"', sql)

        # Remove a trailing semicolon if present.
        sql = re.sub(r";\s*$", "", sql).strip()

        self._cache[cache_key] = sql
        return sql


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert cleaned KPI English explanations to SQL."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(project_root / "data" / "raw" / "extracted_kpis_explained.json"),
        help="Input KPI JSON with pql_explanation field.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "data" / "translation" / "kpis_sql_translation.json"),
        help="Output JSON path with generated sql_query field.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Groq model name.",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env (GROQ_API_KEY=...) or groq_api_key, before running.")

    rows = load_json_rows(Path(args.input))
    translator = SQLTranslator(api_key=api_key, model_name=args.model)

    # Optional context artifacts
    kpi_dep = safe_load_json(project_root / "data" / "processed" / "kpi_dependency_graph" / "kpi_dependency_graph.json")
    table_dep = safe_load_json(project_root / "data" / "processed" / "table_dependency_graph" / "table_dependency_graph.json")
    dm_meta = safe_load_json(project_root / "data" / "raw" / "data_model_metadata.json")

    pairs: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        explanation = str(row.get("pql_explanation", "") or "")
        pql_formula = str(row.get("pql_formula", "") or "")
        kpi_id = str(row.get("kpi_id", "") or "")
        cleaned_explanation = clean_english_explanation(explanation)
        row["pql_explanation_cleaned"] = cleaned_explanation
        extra_context = build_sql_context(
            kpi_id=kpi_id,
            pql_formula=pql_formula,
            kpi_dependency_graph=kpi_dep,
            table_dependency_graph=table_dep,
            data_model_metadata=dm_meta,
        )
        try:
            row["sql_query"] = translator.to_sql(cleaned_explanation, pql_formula, extra_context=extra_context)
            row["sql_translation_status"] = "success"
        except Exception as exc:
            row["sql_query"] = ""
            row["sql_translation_status"] = f"failed: {exc}"

        pairs.append(
            {
                "kpi_id": row.get("kpi_id"),
                "name": row.get("name"),
                "pql_formula": row.get("pql_formula"),
                "sql_query": row.get("sql_query"),
                "sql_translation_status": row.get("sql_translation_status"),
            }
        )

        if idx % 25 == 0:
            print(f"Translated {idx}/{len(rows)} KPI explanations to SQL...")

    output_path = Path(args.output)
    save_json_rows(output_path, rows)
    print(f"Saved SQL translations to: {output_path}")

    # Additionally write a compact PQL -> SQL mapping for easy review.
    pairs_json_path = Path(args.output).parent / "pql_to_sql_pairs.json"
    save_json_rows(pairs_json_path, pairs)

    pairs_txt_path = Path(args.output).parent / "pql_to_sql_pairs.txt"
    save_pairs_report_txt(pairs_txt_path, pairs)
    print(f"Saved PQL->SQL pairs to: {pairs_json_path}")
    print(f"Saved readable report to: {pairs_txt_path}")
    print(
        "LLM token usage (SQL translation): "
        f"input={translator.usage['input_tokens']}, "
        f"output={translator.usage['output_tokens']}, "
        f"total={translator.usage['total_tokens']}"
    )


if __name__ == "__main__":
    main()
