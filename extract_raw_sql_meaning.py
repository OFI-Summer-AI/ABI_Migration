import json
import re
from pathlib import Path
from typing import Dict, List


INPUT_PATH = Path("data/raw/extracted_transformations.json")
OUTPUT_DIR = Path("data/processed/transformation_meaning")


def safe_name(value: str) -> str:
    value = (value or "").strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:150] if len(value) > 150 else value


def unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def extract_meaning(sql: str) -> Dict[str, List[str]]:
    # Support schema-qualified and unqualified names, quoted or unquoted.
    create_pattern = re.compile(
        r'(?is)\bcreate\s+(?:or\s+replace\s+)?(?:table|view)\s+(?:if\s+not\s+exists\s+)?'
        r'(?:'
        r'"[^"]+"\."([^"]+)"'
        r'|'
        r'"([^"]+)"'
        r'|'
        r'([A-Za-z_][A-Za-z0-9_]*)'
        r')'
    )
    drop_pattern = re.compile(
        r'(?is)\bdrop\s+(?:table|view)\s+if\s+exists\s+'
        r'(?:'
        r'"[^"]+"\."([^"]+)"'
        r'|'
        r'"([^"]+)"'
        r'|'
        r'([A-Za-z_][A-Za-z0-9_]*)'
        r')'
    )
    insert_pattern = re.compile(
        r'(?is)\binsert\s+into\s+'
        r'(?:'
        r'"[^"]+"\."([^"]+)"'
        r'|'
        r'"([^"]+)"'
        r'|'
        r'([A-Za-z_][A-Za-z0-9_]*)'
        r')'
    )
    source_pattern = re.compile(
        r'(?is)\b(?:from|join)\s+'
        r'(?:'
        r'"[^"]+"\."([^"]+)"'
        r'|'
        r'"([^"]+)"'
        r'|'
        r'([A-Za-z_][A-Za-z0-9_]*)'
        r')'
    )
    rename_pattern = re.compile(r'(?is)\balter\s+table\s+"[^"]+"\."([^"]+)"\s+rename\s+to\s+([A-Za-z0-9_]+)')
    where_pattern = re.compile(r'(?is)\bwhere\b')
    join_type_pattern = re.compile(r'(?is)\b(left|right|inner|full|cross)\s+join\b')
    commented_out_pattern = re.compile(r'(?is)^\s*/\*.*\*/\s*$')

    # Remove block and line comments before regex extraction to avoid false positives.
    sql_clean = re.sub(r"(?is)/\*.*?\*/", " ", sql)
    sql_clean = re.sub(r"(?im)--.*?$", " ", sql_clean)

    created_objects = unique_keep_order([next((g for g in m if g), "") for m in create_pattern.findall(sql_clean) if any(m)])
    dropped_objects = unique_keep_order([next((g for g in m if g), "") for m in drop_pattern.findall(sql_clean) if any(m)])
    insert_targets = unique_keep_order([next((g for g in m if g), "") for m in insert_pattern.findall(sql_clean) if any(m)])
    source_tables = unique_keep_order([next((g for g in m if g), "") for m in source_pattern.findall(sql_clean) if any(m)])
    rename_targets = [f"{old} -> {new}" for old, new in rename_pattern.findall(sql_clean)]
    join_types = unique_keep_order([x.upper() for x in join_type_pattern.findall(sql_clean)])

    return {
        "is_commented_out_block": [str(bool(commented_out_pattern.match(sql)))],
        "created_objects": created_objects,
        "dropped_objects": dropped_objects,
        "insert_targets": insert_targets,
        "source_tables_or_views": source_tables,
        "renamed_tables": rename_targets,
        "join_types_present": join_types,
        "has_where_clause": [str(bool(where_pattern.search(sql_clean)))],
    }


def build_text_summary(item: dict, meaning: dict) -> str:
    lines = [
        f"Job Name: {item.get('job_name', '')}",
        f"Transformation Name: {item.get('transformation_name', '')}",
        f"Status: {item.get('status', '')}",
        "",
        "Interpreted Meaning",
        "-" * 80,
        f"Commented out block: {meaning['is_commented_out_block'][0]}",
        f"Creates: {', '.join(meaning['created_objects']) if meaning['created_objects'] else 'None'}",
        f"Drops: {', '.join(meaning['dropped_objects']) if meaning['dropped_objects'] else 'None'}",
        f"Inserts into: {', '.join(meaning['insert_targets']) if meaning['insert_targets'] else 'None'}",
        f"Reads from (FROM/JOIN): {', '.join(meaning['source_tables_or_views']) if meaning['source_tables_or_views'] else 'None'}",
        f"Renames: {', '.join(meaning['renamed_tables']) if meaning['renamed_tables'] else 'None'}",
        f"Join types present: {', '.join(meaning['join_types_present']) if meaning['join_types_present'] else 'None'}",
        f"Has WHERE clause: {meaning['has_where_clause'][0]}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sql_dir = OUTPUT_DIR / "raw_sql"
    summary_dir = OUTPUT_DIR / "summaries"
    sql_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    items = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    interpreted = []

    for idx, item in enumerate(items, start=1):
        job = safe_name(item.get("job_name", f"job_{idx}")) or f"job_{idx}"
        transformation = safe_name(item.get("transformation_name", f"transformation_{idx}")) or f"transformation_{idx}"
        stem = f"{idx:03d}__{job}__{transformation}"

        raw_sql = item.get("raw_sql") or ""
        meaning = extract_meaning(raw_sql)

        (sql_dir / f"{stem}.sql").write_text(raw_sql, encoding="utf-8")
        (summary_dir / f"{stem}.txt").write_text(build_text_summary(item, meaning), encoding="utf-8")

        interpreted.append(
            {
                "job_name": item.get("job_name"),
                "transformation_name": item.get("transformation_name"),
                "status": item.get("status"),
                "meaning": meaning,
            }
        )

    (OUTPUT_DIR / "transformations_meaning.json").write_text(
        json.dumps(interpreted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Processed transformations: {len(items)}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Raw SQL files: {sql_dir}")
    print(f"Text summaries: {summary_dir}")
    print(f"JSON summary: {OUTPUT_DIR / 'transformations_meaning.json'}")


if __name__ == "__main__":
    main()
