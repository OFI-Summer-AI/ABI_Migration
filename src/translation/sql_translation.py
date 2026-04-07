import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from config.settings import get_settings
from dependency.kpi_dependency_graph import build_dependency_graph, extract_kpi_refs
from translation.pql_sql_function_reference import (
    default_reference_path,
    load_pql_sql_function_reference,
)


def max_dependency_depth_downstream(
    kpi_id: str,
    graph: Dict[str, List[str]],
    *,
    visiting: Optional[Set[str]] = None,
    memo: Optional[Dict[str, int]] = None,
) -> int:
    """
    Longest chain length from kpi_id through KPI("...") dependencies to a leaf.
    Example: A -> B -> C with C leaf => depth(A)=2, depth(B)=1, depth(C)=0.
    Cycles break with depth 0 for the cycling branch to avoid infinite recursion.
    """
    if memo is None:
        memo = {}
    if kpi_id in memo:
        return memo[kpi_id]
    if visiting is None:
        visiting = set()
    if kpi_id in visiting:
        return 0
    visiting.add(kpi_id)
    deps = [d for d in graph.get(kpi_id, []) if d in graph]
    if not deps:
        memo[kpi_id] = 0
        visiting.remove(kpi_id)
        return 0
    best = 0
    for d in deps:
        best = max(best, 1 + max_dependency_depth_downstream(d, graph, visiting=visiting, memo=memo))
    visiting.remove(kpi_id)
    memo[kpi_id] = best
    return best


def kpi_sql_complexity_metrics(
    kpi_id: str,
    pql: str,
    graph: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Metrics used to decide single-pass vs 2-pass + validation SQL translation."""
    refs = extract_kpi_refs(pql or "")
    known_refs = [r for r in refs if r in graph]
    depth = max_dependency_depth_downstream(kpi_id, graph) if kpi_id in graph else 0
    direct_nested_kpi_count = len(known_refs)
    use_two_pass = depth >= 2 or direct_nested_kpi_count >= 2
    return {
        "kpi_dependency_depth": depth,
        "direct_nested_kpi_count": direct_nested_kpi_count,
        "use_two_pass_sql": use_two_pass,
    }


def build_kpi_graph(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Adjacency: kpi_id -> [referenced KPI ids that exist in the dataset]."""
    g, _ = build_dependency_graph(rows)
    return g


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of KPI objects.")
    # Only KPIs should be translated to SQL; ignore Attributes/Filters if present.
    return [row for row in data if str(row.get("attribute_type", "")).lower() == "kpi"]


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
    """Clean escaped/quoted artifacts before SQL generation prompt."""
    cleaned = (text or "").strip()
    cleaned = cleaned.replace('\\"', '"')
    cleaned = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


SQL_SYSTEM_RULES_SINGLE_PASS = (
    "ROLE\n"
    "- You are a precise transpiler that converts Celonis PQL expressions into SQL.\n"
    "- Treat PQL as the single source of truth for logic.\n"
    "- Use the English KPI explanation only as supporting context when PQL is ambiguous.\n"
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
    "NESTED KPI RULES\n"
    "- PQL may contain KPI(\"child_id\") calls. Do not leave KPI(...) in the output SQL.\n"
    "- Replace each KPI(...) with SQL that matches the described semantics from the English explanation,\n"
    "  while staying faithful to outer PQL structure (aggregations, CASE, filters).\n"
    "- If nested semantics cannot be resolved from the text, use NULL for that subexpression\n"
    "  and add no invented joins.\n"
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
    "- Do not wrap identifiers in double quotes or backticks in final SQL output.\n"
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
    "\n"
    "DATA MODEL CONTEXT\n"
    "- For systematic JOIN keys, rely on the Celonis data model foreign keys (data_model_metadata.json)\n"
    "  and/or table_dependency_graph.json.\n"
    "- Use extracted_transformations.json only as supplemental SQL patterns, not as the primary FK source.\n"
    "- KPI dependency order and nesting: use kpi_dependency_graph.json when planning multi-step KPI logic.\n"
)


def _system_rules_with_reference(function_reference_block: str) -> str:
    block = (function_reference_block or "").strip()
    if not block:
        return SQL_SYSTEM_RULES_SINGLE_PASS
    return (
        SQL_SYSTEM_RULES_SINGLE_PASS
        + "\nMAPPING REFERENCE (PQL → SQL / Spark)\n"
        + "Apply these correspondences when translating constructs below;\n"
        "if a construct is not listed, use standard SQL/Spark semantics.\n\n"
        + block
        + "\n"
    )


class SQLTranslator:
    """Translate plain-English KPI explanations into SQL queries."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        function_reference_block: str = "",
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
        )
        self._system_rules = _system_rules_with_reference(function_reference_block)
        self._cache: Dict[str, str] = {}

    @staticmethod
    def _strip_double_quoted_identifiers(sql: str) -> str:
        """
        Remove double quotes around identifier-like tokens while preserving string literals.
        Examples:
          "VBRK"."VBELN" -> VBRK.VBELN
          "VBELN" -> VBELN
        """
        out = str(sql or "")
        out = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*\.\s*"([A-Za-z_][A-Za-z0-9_]*)"', r"\1.\2", out)
        out = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r"\1", out)
        return out

    @staticmethod
    def _postprocess_sql(sql: str) -> str:
        sql = str(sql or "").strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        sql = re.sub(r"`([^`]+)`", r"\1", sql)
        sql = SQLTranslator._strip_double_quoted_identifiers(sql)
        sql = re.sub(r";\s*$", "", sql).strip()
        return sql

    @staticmethod
    def _validate_sql(sql: str, pql_formula: str) -> List[str]:
        """Return human-readable validation failures; empty list means pass."""
        errors: List[str] = []
        s = (sql or "").strip()
        if not s:
            errors.append("Generated SQL is empty.")
            return errors
        if not re.match(r"(?is)^\s*SELECT\b", s):
            errors.append("SQL must be a single SELECT statement.")
        if re.search(r"(?i)\bKPI\s*\(", s):
            errors.append("SQL must not contain unresolved KPI(...) calls; expand or replace them.")
        if re.search(r"(?is);\s*\S", s):
            errors.append("SQL must not contain multiple statements.")
        return errors

    def _extract_semantic_plan(self, explanation: str, pql_formula: str) -> Dict[str, Any]:
        """Pass 1: structured plan from English + PQL (JSON)."""
        messages = [
            SystemMessage(
                content=(
                    self._system_rules
                    + "\nPASS-1 SEMANTIC EXTRACTION\n"
                    "You analyze Celonis KPI PQL together with its plain-English explanation.\n"
                    "Return valid JSON only (no markdown).\n"
                    "PQL is the source of truth for identifiers and structure.\n"
                    "English explains nested KPI intent where PQL still contains KPI(\"...\") references.\n"
                    "Do not invent tables, columns, or joins.\n"
                    "Required keys:\n"
                    "- outer_structure: string (e.g. ratio, sum, case_when_aggregate, boolean)\n"
                    "- aggregations: array of strings (describe each aggregate applied in order)\n"
                    "- filters_and_conditions: array of strings\n"
                    "- case_when_blocks: array of strings (each major CASE or IF-like construct)\n"
                    "- kpi_nested_references: array of objects with keys kpi_id, role_in_formula\n"
                    "- expression_tree_notes: string (how nested pieces combine)\n"
                    "- fidelity_risks: array of strings (where English vs PQL might disagree)\n"
                )
            ),
            HumanMessage(
                content=(
                    "English explanation:\n"
                    f"{explanation}\n\n"
                    "PQL (source of truth):\n"
                    f"{pql_formula}\n\n"
                    "Emit the JSON plan now."
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
        return {
            "outer_structure": "unknown",
            "aggregations": [],
            "filters_and_conditions": [],
            "case_when_blocks": [],
            "kpi_nested_references": [],
            "expression_tree_notes": text[:2000],
            "fidelity_risks": ["Pass-1 model output was not valid JSON; downstream SQL may be weaker."],
        }

    def _sql_from_plan(
        self,
        explanation: str,
        pql_formula: str,
        semantic_plan: Dict[str, Any],
        *,
        validation_feedback: str = "",
    ) -> str:
        """Pass 2: SQL SELECT from structured plan + PQL + English."""
        feedback_block = (
            f"VALIDATION FAILURES TO FIX (must address each):\n{validation_feedback}\n\n"
            if validation_feedback.strip()
            else ""
        )
        plan_json = json.dumps(semantic_plan, ensure_ascii=False, indent=2)
        messages = [
            SystemMessage(
                content=(
                    self._system_rules
                    + "\nTWO-PASS MODE\n"
                    "- You are given a semantic plan JSON from pass 1 and must emit final SQL.\n"
                    "- Reconcile plan with PQL: if they conflict, follow PQL.\n"
                    "- Nested KPIs: never output KPI(...); encode semantics per English + plan.\n"
                    "- Output one SELECT only, plain text.\n"
                )
            ),
            HumanMessage(
                content=(
                    f"{feedback_block}"
                    "English explanation:\n"
                    f"{explanation}\n\n"
                    "PQL (source of truth):\n"
                    f"{pql_formula}\n\n"
                    "Semantic plan (JSON):\n"
                    f"{plan_json}\n\n"
                    "Generate the equivalent SQL SELECT for this KPI."
                )
            ),
        ]
        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        return self._postprocess_sql(str(content))

    def _single_pass_to_sql(self, normalized_explanation: str, normalized_pql: str) -> str:
        messages = [
            SystemMessage(content=self._system_rules),
            HumanMessage(
                content=(
                    "KPI plain-English explanation (may omit technical details):\n"
                    f"{normalized_explanation}\n\n"
                    "Original PQL (source of truth for semantics):\n"
                    f"{normalized_pql}\n\n"
                    "Generate the equivalent SQL SELECT query for this KPI."
                )
            ),
        ]
        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        return self._postprocess_sql(str(content))

    def to_sql(
        self,
        explanation: str,
        pql_formula: str,
        *,
        use_two_pass: bool = False,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Translate English explanation + PQL to a single SELECT.

        When use_two_pass is True (dependency depth >= 2 or multiple nested KPI refs),
        runs: extract semantic plan -> generate SQL -> validate -> retry with feedback (up to 2 retries).

        If ``diagnostics`` is a dict, it is filled with mode, validation notes, and pass status.
        """
        normalized_explanation = clean_english_explanation(explanation)
        normalized_pql = (pql_formula or "").strip()
        if not normalized_explanation and not normalized_pql:
            return "-- Missing explanation and PQL; SQL cannot be generated."

        mode = "two_pass" if use_two_pass else "single_pass"
        cache_key = f"{mode}||{normalized_pql}||{normalized_explanation}"
        if cache_key in self._cache:
            cached_sql = self._cache[cache_key]
            if diagnostics is not None:
                diagnostics["sql_translation_mode"] = mode
                diagnostics["from_cache"] = True
                diagnostics["sql_validation_ok"] = len(self._validate_sql(cached_sql, normalized_pql)) == 0
                diagnostics["sql_validation_notes"] = []
            return cached_sql

        if diagnostics is not None:
            diagnostics["sql_translation_mode"] = mode
            diagnostics["from_cache"] = False

        if not use_two_pass:
            sql = self._single_pass_to_sql(normalized_explanation, normalized_pql)
            self._cache[cache_key] = sql
            if diagnostics is not None:
                diagnostics["sql_validation_ok"] = len(self._validate_sql(sql, normalized_pql)) == 0
                diagnostics["sql_validation_notes"] = []
            return sql

        plan = self._extract_semantic_plan(normalized_explanation, normalized_pql)
        validation_notes: List[str] = []
        sql = ""
        feedback = ""
        max_attempts = 3
        final_ok = False
        for attempt in range(max_attempts):
            sql = self._sql_from_plan(
                normalized_explanation,
                normalized_pql,
                plan,
                validation_feedback=feedback,
            )
            errs = self._validate_sql(sql, normalized_pql)
            if not errs:
                final_ok = True
                break
            validation_notes.append(f"attempt {attempt + 1}: " + "; ".join(errs))
            feedback = "\n".join(f"- {e}" for e in errs)
        self._cache[cache_key] = sql
        if diagnostics is not None:
            diagnostics["sql_validation_ok"] = final_ok
            diagnostics["sql_validation_notes"] = validation_notes
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
        default="gpt-4o-mini",
        help="OpenAI chat model name (e.g. gpt-4o-mini, gpt-4o).",
    )
    parser.add_argument(
        "--pql-sql-reference",
        type=str,
        default="",
        help="Path to PQL_to_SQL_Function_Reference.xlsx (default: data/raw/PQL_to_SQL_Function_Reference.xlsx).",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env (or openai_api_key) before running.")

    ref_path = Path(args.pql_sql_reference) if args.pql_sql_reference.strip() else default_reference_path(project_root)
    func_ref = load_pql_sql_function_reference(ref_path)
    if func_ref:
        print(f"Loaded PQL→SQL function reference: {ref_path}")
    else:
        print(f"No function reference loaded (missing or empty): {ref_path}")

    rows = load_json_rows(Path(args.input))
    translator = SQLTranslator(
        api_key=api_key,
        model_name=args.model,
        function_reference_block=func_ref,
    )
    graph = build_kpi_graph(rows)

    pairs: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        explanation = str(row.get("pql_explanation", "") or "")
        pql_formula = str(row.get("pql_formula", "") or "")
        cleaned_explanation = clean_english_explanation(explanation)
        row["pql_explanation_cleaned"] = cleaned_explanation
        expanded_pql = str(row.get("pql_formula_expanded", "") or "").strip()
        if expanded_pql:
            cleaned_explanation_for_sql = (
                f"{cleaned_explanation}\n\nExpanded PQL context:\n{expanded_pql}"
            ).strip()
        else:
            cleaned_explanation_for_sql = cleaned_explanation

        kpi_id = str(row.get("kpi_id", "") or "").strip()
        metrics = kpi_sql_complexity_metrics(kpi_id, pql_formula, graph)
        row["kpi_dependency_depth"] = metrics["kpi_dependency_depth"]
        row["direct_nested_kpi_count"] = metrics["direct_nested_kpi_count"]
        row["use_two_pass_sql"] = metrics["use_two_pass_sql"]

        diag: Dict[str, Any] = {}
        try:
            row["sql_query"] = translator.to_sql(
                cleaned_explanation_for_sql,
                pql_formula,
                use_two_pass=metrics["use_two_pass_sql"],
                diagnostics=diag,
            )
            row["sql_translation_mode"] = diag.get("sql_translation_mode", "")
            row["sql_validation_ok"] = diag.get("sql_validation_ok", True)
            row["sql_validation_notes"] = diag.get("sql_validation_notes", [])
            row["sql_translation_status"] = "success"
        except Exception as exc:
            row["sql_query"] = ""
            row["sql_translation_status"] = f"failed: {exc}"
            row["sql_translation_mode"] = ""
            row["sql_validation_ok"] = False
            row["sql_validation_notes"] = [str(exc)]

        pairs.append(
            {
                "kpi_id": row.get("kpi_id"),
                "name": row.get("name"),
                "pql_formula": row.get("pql_formula"),
                "sql_query": row.get("sql_query"),
                "sql_translation_status": row.get("sql_translation_status"),
                "use_two_pass_sql": row.get("use_two_pass_sql"),
                "kpi_dependency_depth": row.get("kpi_dependency_depth"),
                "sql_translation_mode": row.get("sql_translation_mode"),
                "sql_validation_ok": row.get("sql_validation_ok"),
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


if __name__ == "__main__":
    main()
