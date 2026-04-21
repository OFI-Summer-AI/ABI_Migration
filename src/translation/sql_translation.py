"""
PQL → SQL translator using a bottom-up (dependency-first) strategy.

Key design decisions
--------------------
1. KPI references are resolved before calling the LLM.
   KPI("some-id", ...) inside a formula is replaced with the SQL that was
   already generated for that dependency in a prior iteration.  The LLM
   therefore never sees opaque KPI("…") tokens — it only sees concrete SQL.

2. Translation order follows the topological sort of the dependency graph
   (leaves first, root KPIs last).  This guarantees every dependency's SQL
   is available when its parent is processed.

3. The system prompt contains an explicit PQL-function-to-SQL mapping table
   so the model has domain knowledge of every Celonis-specific construct.

4. {p1}, {p2}, … template parameters are converted to SQL named parameters
   (:p1, :p2, …) and explained in the prompt.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Detects KPI ID from any KPI() call — closing ')' NOT required so that
# parameterised calls like KPI("id",{p1},{p2}) are still captured.
KPI_CALL_RE = re.compile(
    r"""(?ix)
    \b KPI \s* \( \s*
      (?:
        "([^"]+)"
        | '([^']+)'
      )
    """
)

# Matches the FULL KPI() call (including optional extra params) for replacement.
# Handles: KPI("id"), KPI("id",{p1},{p2}), KPI('id',{p1}).
# Assumes extra parameters don't contain unbalanced ')' characters (true for {…} placeholders).
KPI_FULL_CALL_RE = re.compile(
    r"""(?ix)
    \b KPI \s* \( \s*
      (?:"[^"]+" | '[^']+')                                  # first arg: quoted KPI id
      (?:\s*,\s*(?:\{[^}]*\}|'[^']*'|"[^"]*"|[^,)]+))*     # optional extra params
    \s* \)
    """
)

TABLE_COL_RE = re.compile(r'"([^"]+)"\."([^"]+)"')

# ---------------------------------------------------------------------------
# PQL function → SQL mapping (used in the system prompt)
# ---------------------------------------------------------------------------
PQL_FUNCTION_MAP = """
PQL FUNCTION → SQL MAPPING REFERENCE
======================================
Use this table as the authoritative reference when translating any PQL expression.

| PQL construct                              | SQL equivalent / notes                                                      |
|--------------------------------------------|-----------------------------------------------------------------------------|
| COUNT_TABLE("T")                           | (SELECT COUNT(*) FROM "T")                                                  |
| COUNT(expr)                                | COUNT(expr)                                                                 |
| COUNT(DISTINCT expr)                       | COUNT(DISTINCT expr)                                                        |
| SUM(expr)                                  | SUM(expr)                                                                   |
| AVG(expr)                                  | AVG(expr)                                                                   |
| MIN(expr) / MAX(expr)                      | MIN(expr) / MAX(expr)                                                       |
| CASE WHEN c THEN v ELSE d END              | CASE WHEN c THEN v ELSE d END  (identical)                                  |
| COALESCE(a, b)                             | COALESCE(a, b)                                                              |
| "TABLE"."COLUMN"                           | "TABLE"."COLUMN"  (keep double-quote style)                                 |
| a || b  (string concat)                    | CONCAT(a, b)  or  a || b  (dialect-dependent)                               |
| SUBSTRING(str, start, len)                 | SUBSTRING(str FROM start+1 FOR len)  — PQL is 0-indexed, SQL is 1-indexed  |
| LENGTH(str)                                | LENGTH(str)  or  CHAR_LENGTH(str)                                           |
| TO_FLOAT(x)                                | CAST(x AS FLOAT)                                                            |
| TO_INT(x)                                  | CAST(x AS BIGINT)                                                           |
| TO_STRING(x)                               | CAST(x AS VARCHAR)                                                          |
| ROUND(x, n)                                | ROUND(x, n)                                                                 |
| ABS(x)                                     | ABS(x)                                                                      |
| FLOOR(x) / CEIL(x)                         | FLOOR(x) / CEIL(x)                                                          |
| DATEDIFF(unit, d1, d2)                     | DATEDIFF(unit, d1, d2)  or  TIMESTAMPDIFF depending on dialect              |
| TODAY()                                    | CURRENT_DATE                                                                |
| NOW()                                      | CURRENT_TIMESTAMP                                                           |
| DATEFORMAT(d, fmt)                         | TO_CHAR(d, fmt)  (Snowflake/Postgres) or DATE_FORMAT(d, fmt) (MySQL)        |
| YEAR(d) / MONTH(d) / DAY(d)               | EXTRACT(YEAR FROM d) / EXTRACT(MONTH FROM d) / EXTRACT(DAY FROM d)         |
| REMAP_INTS(col,[from1,from2],[to1,to2])    | CASE WHEN col=from1 THEN to1 WHEN col=from2 THEN to2 … END                  |
| PROCESS_ORDER(tbl, event_col, case_col)    | ROW_NUMBER() OVER (PARTITION BY case_col ORDER BY event_col)               |
| VARIANT(col)                               | STRING_AGG(col, ' → ' ORDER BY col)  (approximate; mark with a comment)     |
| SHORTENED(expr)                            | LEFT(CAST(expr AS VARCHAR), 500)  (truncates long strings)                  |
| BIND(tbl, col)                             | /* BIND: correlated join between tbl and the current case table */          |
|                                            | Emit the column reference as "tbl"."col" and add a join comment if needed.  |
| INDEX_AVG / INDEX_MEDIAN                   | AVG / PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY …)                        |
| PU_AVG(tbl, expr)                          | AVG(expr) aggregated at the granularity of tbl (use a subquery/CTE)         |
| PU_SUM(tbl, expr)                          | SUM(expr) aggregated at the granularity of tbl (use a subquery/CTE)         |
| PU_COUNT(tbl, expr)                        | COUNT(expr) aggregated at the granularity of tbl (use a subquery/CTE)       |
| PU_FIRST / PU_LAST                         | FIRST_VALUE / LAST_VALUE window function                                    |
| RUNNING_SUM / RUNNING_AVG                  | SUM(...) OVER (...) / AVG(...) OVER (...)                                   |
| {p1}, {p2}, …                              | Named SQL parameters :p1, :p2, … (runtime filter values injected by caller) |
| KPI("id")                                  | Already replaced by the caller with the SQL for that KPI before you see it. |
|                                            | If you still see KPI("id"), treat it as an unresolved subquery placeholder. |
"""

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_pairs_report_txt(path: Path, pairs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for i, pair in enumerate(pairs, start=1):
        lines.append(f"[{i}] KPI_ID: {pair.get('kpi_id','')} | NAME: {pair.get('name','')}".rstrip())
        lines.append("PQL:")
        lines.append(str(pair.get("pql_formula", "") or ""))
        lines.append("PQL (with inlined deps):")
        lines.append(str(pair.get("pql_inlined", "") or ""))
        lines.append("SQL:")
        lines.append(str(pair.get("sql_query", "") or ""))
        lines.append("-" * 60)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def extract_kpi_refs(pql: str) -> List[str]:
    refs: List[str] = []
    for m in KPI_CALL_RE.finditer(pql or ""):
        ref = (m.group(1) or m.group(2) or "").strip()
        if ref:
            refs.append(ref)
    return refs


def extract_tables_from_pql(pql: str) -> List[str]:
    return sorted({t for t, _ in TABLE_COL_RE.findall(pql or "")})


def substitute_template_params(pql: str) -> str:
    """Replace Celonis {p1},{p2},... template params with SQL named params :p1,:p2,..."""
    return re.sub(r"\{(\w+)\}", r":\1", pql)


def inline_kpi_sql(pql: str, sql_cache: Dict[str, str]) -> str:
    """Replace every KPI("id",...) call with the already-translated SQL for that KPI.

    Uses KPI_FULL_CALL_RE to match the complete call (including extra params),
    then extracts the KPI ID via KPI_CALL_RE and looks it up in sql_cache.
    Falls back to keeping the original token when the ID is not in the cache.
    """
    def repl(match: re.Match) -> str:
        full_call = match.group(0)
        id_match = KPI_CALL_RE.search(full_call)
        if not id_match:
            return full_call
        ref = (id_match.group(1) or id_match.group(2) or "").strip()
        cached = sql_cache.get(ref)
        if cached:
            return f"({cached})"
        return full_call  # keep as-is when not yet translated (shouldn't happen in topo order)

    return KPI_FULL_CALL_RE.sub(repl, pql)


def build_kpi_graph(kpi_index: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build adjacency list: kpi_id -> [dependency_kpi_ids]."""
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
    """Leaf-first topological sort.  Handles cycles gracefully."""
    remaining = {n: set(deps) for n, deps in graph.items()}
    order: List[str] = []
    while remaining:
        ready = sorted([n for n, deps in remaining.items() if not deps])
        if not ready:
            # Cycle detected — break it by picking the node with fewest remaining deps.
            ready = [min(remaining, key=lambda n: len(remaining[n]))]
        for n in ready:
            order.append(n)
            del remaining[n]
            for deps in remaining.values():
                deps.discard(n)
    return order


def clean_english_explanation(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace('\\"', '"')
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\[\[.*?\]\]", " ", cleaned)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned)
    cleaned = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def build_sql_context(
    *,
    kpi_id: str,
    pql_formula: str,
    kpi_dependency_graph: Any,
    table_dependency_graph: Any,
    data_model_metadata: Any,
) -> str:
    parts: List[str] = []

    if isinstance(data_model_metadata, dict):
        ref_tables = set(extract_tables_from_pql(pql_formula))
        if ref_tables:
            fk_lines: List[str] = []
            for fk in data_model_metadata.get("foreign_keys") or []:
                s = fk.get("source_table_name")
                t = fk.get("target_table_name")
                if s in ref_tables and t in ref_tables:
                    fk_lines.append(f"{s} → {t}")
            if fk_lines:
                parts.append("Relevant foreign keys: " + ", ".join(fk_lines[:20]))

    if isinstance(table_dependency_graph, dict):
        ref_tables = set(extract_tables_from_pql(pql_formula))
        if ref_tables:
            edges = table_dependency_graph.get("edges") or []
            edge_lines = [
                f"{e.get('from')} → {e.get('to')}"
                for e in edges
                if e.get("from") in ref_tables and e.get("to") in ref_tables
            ]
            if edge_lines:
                parts.append("Relevant table relationships: " + ", ".join(edge_lines[:20]))

    return "\n".join(parts)[:4000]


# ---------------------------------------------------------------------------
# SQL Translator
# ---------------------------------------------------------------------------

class SQLTranslator:
    """Translate PQL expressions (with dependencies already inlined) into SQL."""

    SYSTEM_PROMPT = (
        "ROLE\n"
        "You are an expert SQL engineer and Celonis PQL specialist. "
        "Your task is to convert a Celonis PQL expression into a correct, "
        "executable SQL SELECT statement.\n"
        "\n"
        "INPUTS YOU RECEIVE\n"
        "1. 'Original PQL' — the raw PQL formula as written in Celonis.\n"
        "2. 'Inlined PQL' — the same formula with every KPI(\"id\") reference "
        "replaced by the SQL for that dependency.  Use this as the primary "
        "source of truth for the SQL logic.\n"
        "3. 'English description' — a plain-English summary; treat as "
        "supporting context only.\n"
        "4. Optional table/FK context.\n"
        "\n"
        "MANDATORY OUTPUT CONTRACT\n"
        "1. Return exactly one SQL SELECT statement — nothing else.\n"
        "2. Plain SQL text only: no markdown fences, no comments, no explanation.\n"
        "3. No trailing semicolon.\n"
        "4. No DDL/DML (CREATE/ALTER/DROP/INSERT/UPDATE/DELETE/MERGE).\n"
        "\n"
        "SEMANTIC PRESERVATION RULES\n"
        "- Preserve arithmetic, operator precedence, CASE logic, and DISTINCT exactly.\n"
        "- Do not add or remove predicates, aggregations, or joins not implied by PQL.\n"
        "- Preserve null behaviour; do not add COALESCE/IFNULL unless PQL implies it.\n"
        "- {p1},{p2},… become SQL named params :p1,:p2,… — keep them in the output.\n"
        "- Use double-quoted identifiers (\"TABLE\".\"COLUMN\") — never backticks.\n"
        "- For CASE WHEN … THEN 1 ELSE 0 patterns used inside SUM/COUNT, "
        "keep the indicator pattern as-is (do not simplify to COUNT(CASE …)).\n"
        "\n"
        + PQL_FUNCTION_MAP
        + "\n"
        "QUALITY CHECK\n"
        "- Every expression in your output must be traceable to the Inlined PQL.\n"
        "- The output must be syntactically valid SQL.\n"
        "- If a construct cannot be translated faithfully, emit NULL AS <col_name> "
        "for that expression and keep everything else.\n"
    )

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

    def to_sql(
        self,
        explanation: str,
        original_pql: str,
        inlined_pql: str,
        extra_context: str = "",
    ) -> str:
        normalized_original = (original_pql or "").strip()
        normalized_inlined = (inlined_pql or "").strip()
        if not normalized_inlined and not normalized_original:
            return "-- Missing PQL; SQL cannot be generated."

        cache_key = normalized_inlined or normalized_original
        if cache_key in self._cache:
            return self._cache[cache_key]

        context_block = (
            f"Additional table/FK context:\n{extra_context}\n\n"
            if extra_context
            else ""
        )

        human_content = (
            f"Original PQL:\n{normalized_original}\n\n"
            f"Inlined PQL (KPI references already substituted with SQL):\n{normalized_inlined}\n\n"
            f"English description:\n{clean_english_explanation(explanation)}\n\n"
            f"{context_block}"
            "Generate the equivalent SQL SELECT query."
        )

        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])
        self._record_usage(response)

        content = response.content
        if isinstance(content, list):
            content = " ".join(str(p) for p in content)

        sql = str(content).strip()
        # Strip any markdown fences the model might emit despite instructions.
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        sql = re.sub(r"`([^`]+)`", r'"\1"', sql)   # backtick → double-quote
        sql = re.sub(r";\s*$", "", sql).strip()     # remove trailing semicolon

        self._cache[cache_key] = sql
        return sql


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert KPI PQL formulas to SQL (bottom-up).")
    parser.add_argument(
        "--input",
        type=str,
        default=str(project_root / "data" / "raw" / "extracted_kpis_explained.json"),
        help="Input KPI JSON (should contain pql_explanation field from semantic_translation).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "data" / "translation" / "kpis_sql_translation.json"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o).",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env before running.")

    rows = load_json_rows(Path(args.input))

    # Build KPI index and dependency graph for bottom-up ordering.
    kpi_index: Dict[str, Dict[str, Any]] = {
        str(row.get("kpi_id", "")): row
        for row in rows
        if row.get("kpi_id")
    }
    graph = build_kpi_graph(kpi_index)
    topo_order = topo_sort_bottom_up(graph)

    # Include any KPIs not captured in the graph (isolated / no deps).
    in_graph = set(topo_order)
    for kpi_id in kpi_index:
        if kpi_id not in in_graph:
            topo_order.append(kpi_id)

    # Optional context artifacts.
    kpi_dep = safe_load_json(
        project_root / "data" / "processed" / "kpi_dependency_graph" / "kpi_dependency_graph.json"
    )
    table_dep = safe_load_json(
        project_root / "data" / "processed" / "table_dependency_graph" / "table_dependency_graph.json"
    )
    dm_meta = safe_load_json(project_root / "data" / "raw" / "data_model_metadata.json")

    translator = SQLTranslator(api_key=api_key, model_name=args.model)
    sql_cache: Dict[str, str] = {}  # kpi_id -> generated SQL (used to inline into dependents)

    total = len(topo_order)
    for idx, kpi_id in enumerate(topo_order, start=1):
        row = kpi_index.get(kpi_id)
        if not row:
            continue

        original_pql = str(row.get("pql_formula", "") or "")
        explanation = str(row.get("pql_explanation", "") or "")

        # Step 1: inline already-translated SQL for every KPI() reference in the formula.
        inlined_pql = inline_kpi_sql(original_pql, sql_cache)

        # Step 2: replace {p1},{p2},... with SQL named params :p1,:p2,...
        inlined_pql = substitute_template_params(inlined_pql)

        # Step 3: build table/FK context from the inlined formula (richer than original).
        extra_context = build_sql_context(
            kpi_id=kpi_id,
            pql_formula=inlined_pql,
            kpi_dependency_graph=kpi_dep,
            table_dependency_graph=table_dep,
            data_model_metadata=dm_meta,
        )

        try:
            sql = translator.to_sql(
                explanation,
                original_pql=original_pql,
                inlined_pql=inlined_pql,
                extra_context=extra_context,
            )
            row["sql_query"] = sql
            row["pql_inlined"] = inlined_pql
            row["sql_translation_status"] = "success"
            sql_cache[kpi_id] = sql  # cache so dependents can inline it
        except Exception as exc:
            row["sql_query"] = ""
            row["pql_inlined"] = inlined_pql
            row["sql_translation_status"] = f"failed: {exc}"

        if idx % 25 == 0:
            print(f"Translated {idx}/{total} KPIs...")

    # Restore original list order for output.
    original_order = {str(row.get("kpi_id", "")): i for i, row in enumerate(rows)}
    rows.sort(key=lambda r: original_order.get(str(r.get("kpi_id", "") or ""), 999999))

    output_path = Path(args.output)
    save_json_rows(output_path, rows)
    print(f"Saved SQL translations to: {output_path}")

    # Compact PQL → SQL pair report.
    pairs = [
        {
            "kpi_id": row.get("kpi_id"),
            "name": row.get("name"),
            "pql_formula": row.get("pql_formula"),
            "pql_inlined": row.get("pql_inlined"),
            "sql_query": row.get("sql_query"),
            "sql_translation_status": row.get("sql_translation_status"),
        }
        for row in rows
    ]
    pairs_json = output_path.parent / "pql_to_sql_pairs.json"
    pairs_txt = output_path.parent / "pql_to_sql_pairs.txt"
    save_json_rows(pairs_json, pairs)
    save_pairs_report_txt(pairs_txt, pairs)
    print(f"Saved PQL→SQL pairs to: {pairs_json}")
    print(f"Saved readable report to: {pairs_txt}")
    print(
        "LLM token usage (SQL translation): "
        f"input={translator.usage['input_tokens']}, "
        f"output={translator.usage['output_tokens']}, "
        f"total={translator.usage['total_tokens']}"
    )


if __name__ == "__main__":
    main()
