import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings


class PQLExplainer:
    """Translate PQL formulas into plain-English business explanations."""

    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-20b"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0,
            groq_api_key=api_key,
        )
        self._cache: Dict[str, str] = {}

    def explain(self, pql_formula: str) -> str:
        """Return a concise, plain-English explanation for one PQL formula."""
        normalized = (pql_formula or "").strip()
        if not normalized:
            return "No PQL formula available."

        if normalized in self._cache:
            return self._cache[normalized]

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
        self._cache[normalized] = explanation
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

    for idx, kpi in enumerate(kpis, start=1):
        pql_formula = str(kpi.get("pql_formula", "") or "")
        try:
            kpi["pql_explanation"] = explainer.explain(pql_formula)
        except Exception as exc:
            # Keep processing remaining KPIs; attach a fallback message per failed row.
            kpi["pql_explanation"] = f"Explanation unavailable: {exc}"
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(kpis)} KPIs...")

    save_kpis(output_path, kpis)
    print(f"Saved explained KPIs to: {output_path}")


if __name__ == "__main__":
    main()
